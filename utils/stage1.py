import argparse, os, datetime, time
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision import datasets

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.ImbalanceImageNet import LT_Dataset
from datasets.tinyimages_300k import TinyImages
from models.resnet import ResNet18, ResNet34
from models.resnet_imagenet import ResNet50

from utils.utils import *
from utils.ltr_metrics import *
from utils.loss_fn import *

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='PASCL for OOD detection in long-tailed recognition')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--num_workers', '--cpus', type=int, default=64, help='number of threads for data loader')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets', help='data root path')
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--id_class_number', type=int, default=1000, help='for ImageNet subset')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'ResNet50'], help='which model to use')
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    # training params:
    parser.add_argument('--batch_size', '-b', type=int, default=256, help='input batch size for training')
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000, help='input batch size for testing')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay_epochs', '--de', default=[60,80], nargs='+', type=int, help='milestones for multisteps lr decay')
    parser.add_argument('--opt', default='adam', choices=['sgd', 'adam'], help='which optimizer to use')
    parser.add_argument('--decay', default='cos', choices=['cos', 'multisteps'], help='which lr decay method to use')
    parser.add_argument('--Lambda', default=0.5, type=float, help='OE loss term tradeoff hyper-parameter')
    parser.add_argument('--Lambda2', default=0.1, type=float, help='Contrastive loss term tradeoff hyper-parameter')
    parser.add_argument('--T', default=0.07, type=float, help='Temperature in NT-Xent loss (contrastive loss)')
    parser.add_argument('--k', default=0.4, type=float, help='bottom-k classes are taken as tail class')
    parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
    # 
    parser.add_argument('--timestamp', action='store_true', help='If true, attack time stamp after exp str')
    parser.add_argument('--resume', action='store_true', help='If true, resume from early stopped ckpt')
    parser.add_argument('--save_root_path', '--srp', default='/ssd1/haotao/', help='data root path')
    # ddp 
    parser.add_argument('--ddp', action='store_true', help='If true, use distributed data parallel')
    parser.add_argument('--ddp_backend', '--ddpbed', default='nccl', choices=['nccl', 'gloo', 'mpi'], help='If true, use distributed data parallel')
    parser.add_argument('--num_nodes', default=1, type=int, help='Number of nodes')
    parser.add_argument('--node_id', default=0, type=int, help='Node ID')
    parser.add_argument('--dist_url', default='tcp://localhost:23456', type=str, help='url used to set up distributed training')
    args = parser.parse_args()

    assert args.k>0, "When args.k==0, it is just the OE baseline."

    if args.dataset == 'imagenet':
        # adjust learning rate:
        args.lr *= args.batch_size / 256. # linearly scaled to batch size

    return args


def create_save_path():
    # mkdirs:
    decay_str = args.decay
    if args.decay == 'multisteps':
        decay_str += '-'.join(map(str, args.decay_epochs)) 
    opt_str = args.opt 
    if args.opt == 'sgd':
        opt_str += '-m%s' % args.momentum
    opt_str = 'e%d-b%d-%s-lr%s-wd%s-%s' % (args.epochs, args.batch_size, opt_str, args.lr, args.wd, decay_str)
    reweighting_fn_str = 'sign' 
    loss_str = 'Lambda%s-Lambda2%s-T%s-%s' % (args.Lambda, args.Lambda2, args.T, reweighting_fn_str)
    loss_str += '-k%s'% (args.k)
    exp_str = '%s_%s' % (opt_str, loss_str)
    if args.timestamp:
        exp_str += '_%s' % datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    dataset_str = '%s-%s-OOD%d' % (args.dataset, args.imbalance_ratio, args.num_ood_samples) if 'imagenet' not in args.dataset else '%s%d-lt' % (args.dataset, args.id_class_number)
    save_dir = os.path.join(args.save_root_path, 'LT_OOD_results', 'PASCL', dataset_str, args.model, exp_str)
    create_dir(save_dir)
    print('Saving to %s' % save_dir)

    return save_dir

def setup(rank, ngpus_per_node, args):
    # initialize the process group
    world_size = ngpus_per_node * args.num_nodes
    dist.init_process_group(args.ddp_backend, init_method=args.dist_url, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(gpu_id, ngpus_per_node, args): 

    save_dir = args.save_dir

    # get globale rank (thread id):
    rank = args.node_id * ngpus_per_node + gpu_id

    print(f"Running on rank {rank}.")

    # Initializes ddp:
    if args.ddp:
        setup(rank, ngpus_per_node, args)

    # intialize device:
    device = gpu_id if args.ddp else 'cuda'
    torch.backends.cudnn.benchmark = True

    # get batch size:
    train_batch_size = args.batch_size if not args.ddp else int(args.batch_size/ngpus_per_node/args.num_nodes)
    num_workers = args.num_workers if not args.ddp else int((args.num_workers+ngpus_per_node)/ngpus_per_node)

    # data:
    if args.dataset in ['cifar10', 'cifar100']:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    elif args.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=TwoCropTransform(train_transform), imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
    elif args.dataset == 'imagenet':
        num_classes = args.id_class_number
        train_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=TwoCropTransform(train_transform), 
            subset_class_idx=np.arange(0,args.id_class_number))
        test_set = LT_Dataset(
            os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_val.txt', transform=test_transform,
            subset_class_idx=np.arange(0,args.id_class_number))
    if args.ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=num_workers, 
                                drop_last=False, pin_memory=True)
    if args.dataset in ['cifar10', 'cifar100']:
        ood_set = Subset(TinyImages(args.data_root_path, transform=train_transform), list(range(args.num_ood_samples)))
    elif args.dataset == 'imagenet':
        ood_set = datasets.ImageFolder(os.path.join(args.data_root_path, 'imagenet_extra_1k'), transform=train_transform)
    if args.ddp:
        ood_sampler = torch.utils.data.distributed.DistributedSampler(ood_set)
    else:
        ood_sampler = None
    ood_loader = DataLoader(ood_set, batch_size=train_batch_size, shuffle=not args.ddp, num_workers=num_workers,
                                drop_last=True, pin_memory=True, sampler=ood_sampler)
    print('Training on %s with %d images and %d validation images | %d OOD training images.' % (args.dataset, len(train_set), len(test_set), len(ood_set)))
    
    # get prior distributions:
    img_num_per_cls = np.array(train_set.img_num_per_cls)
    img_num_per_cls = torch.from_numpy(img_num_per_cls).to(device)

    _sigmoid_x = torch.linspace(-1, 1, num_classes).to(device)
    _d = -2 * args.k + 1 - 0.001 # - 0.001 to make _d<-1 when k=1
    cl_loss_weights = torch.sign((_sigmoid_x-_d))
    plt.plot(cl_loss_weights.detach().cpu().numpy())
    plt.grid(True, which='both')
    plt.savefig(os.path.join(save_dir, 'cl_loss_weights.png'))
    plt.close()

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes, return_features=True).to(device)
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes, return_features=True).to(device)
    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes, return_features=True).to(device)
    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], broadcast_buffers=False)
    else:
        # model = torch.nn.DataParallel(model)
        pass

    # optimizer:
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum, nesterov=True)
    if args.decay == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.decay == 'multisteps':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_epochs, gamma=0.1)

    # train:
    if args.resume:
        ckpt = torch.load(os.path.join(save_dir, 'latest.pth'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])  
        start_epoch = ckpt['epoch']+1 
        best_overall_acc = ckpt['best_overall_acc']
        training_losses = ckpt['training_losses']
        test_clean_losses = ckpt['test_clean_losses']
        f1s = ckpt['f1s']
        overall_accs = ckpt['overall_accs']
        many_accs = ckpt['many_accs']
        median_accs = ckpt['median_accs']
        low_accs = ckpt['low_accs']
    else:
        training_losses, test_clean_losses = [], []
        f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
        best_overall_acc = 0
        start_epoch = 0

    fp = open(os.path.join(save_dir, 'train_log.txt'), 'a+')
    fp_val = open(os.path.join(save_dir, 'val_log.txt'), 'a+')
    for epoch in range(start_epoch, args.epochs):
        # reset sampler when using ddp:
        if args.ddp:
            train_sampler.set_epoch(epoch)
        start_time = time.time()

        model.train()
        training_loss_meter = AverageMeter()
        current_lr = scheduler.get_last_lr()
        for batch_idx, ((in_data, labels), (ood_data, _)) in enumerate(zip(train_loader, ood_loader)):
            in_data = torch.cat([in_data[0], in_data[1]], dim=0) # shape=(2*N,C,H,W). Two views of each image.
            in_data, labels = in_data.to(device), labels.to(device)
            ood_data = ood_data.to(device)

            N_in = labels.shape[0]

            all_data = torch.cat([in_data, ood_data], dim=0) # shape=(2*Nin+Nout,C,W,H)

            # forward:
            all_logits, p4 = model(all_data)
            in_logits = all_logits[0:2*N_in]
            in_loss = F.cross_entropy(in_logits, torch.cat([labels, labels], dim=0))

            ood_logits = all_logits[2*N_in:]
            ood_loss = oe_loss_fn(ood_logits)

            # contrastive loss between tail-class and OOD samples:
            tail_idx = labels>= round((1-args.k)*num_classes) # dont use int! since 1-0.9=0.0999!=0.1
            if args.ddp:
                all_f = model.module.forward_projection(p4)
            else:
                all_f = model.forward_projection(p4)
            f_id_view1, f_id_view2 = all_f[0:N_in], all_f[N_in:2*N_in]
            f_id_tail_view1 = f_id_view1[tail_idx] # i.e., 6,7,8,9 in cifar10
            f_id_tail_view2 = f_id_view2[tail_idx] # i.e., 6,7,8,9 in cifar10
            labels_tail = labels[tail_idx]
            f_ood = all_f[2*N_in:]
            if torch.sum(tail_idx) > 0:
                cl_loss = my_cl_loss_fn3(
                    torch.stack((f_id_tail_view1, f_id_tail_view2), dim=1), f_ood, labels_tail, temperature=args.T,
                    reweighting=True, w_list=cl_loss_weights
                )
            else:
                cl_loss = 0*ood_loss

            loss = in_loss + args.Lambda * ood_loss + args.Lambda2 * cl_loss

            # backward:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # append:
            training_loss_meter.append(loss.item())
            if rank == 0 and batch_idx % 100 == 0:
                train_str = 'epoch %d batch %d (train): loss %.4f (%.4f, %.4f, %.4f) | lr %s' % (
                    epoch, batch_idx, loss.item(), in_loss.item(), ood_loss.item(), cl_loss.item(), current_lr) 
                print(train_str)
                fp.write(train_str + '\n')
                fp.flush()

        # lr update:
        scheduler.step()

        if rank == 0:
            # eval on clean set:
            model.eval()
            test_acc_meter, test_loss_meter = AverageMeter(), AverageMeter()
            preds_list, labels_list = [], []
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)
                    logits, _ = model(data)
                    pred = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    loss = F.cross_entropy(logits, labels)
                    test_acc_meter.append((logits.argmax(1) == labels).float().mean().item())
                    test_loss_meter.append(loss.item())
                    preds_list.append(pred)
                    labels_list.append(labels)

            preds = torch.cat(preds_list, dim=0).detach().cpu().numpy().squeeze()
            labels = torch.cat(labels_list, dim=0).detach().cpu().numpy()

            overall_acc= (preds == labels).sum().item() / len(labels)
            f1 = f1_score(labels, preds, average='macro')

            many_acc, median_acc, low_acc, _ = shot_acc(preds, labels, img_num_per_cls, acc_per_cls=True)

            test_clean_losses.append(test_loss_meter.avg)
            f1s.append(f1)
            overall_accs.append(overall_acc)
            many_accs.append(many_acc)
            median_accs.append(median_acc)
            low_accs.append(low_acc)

            val_str = 'epoch %d (test): ACC %.4f (%.4f, %.4f, %.4f) | F1 %.4f | time %s' % (epoch, overall_acc, many_acc, median_acc, low_acc, f1, time.time()-start_time) 
            print(val_str)
            fp_val.write(val_str + '\n')
            fp_val.flush()

            # save curves:
            training_losses.append(training_loss_meter.avg)
            plt.plot(training_losses, 'b', label='training_losses')
            plt.plot(test_clean_losses, 'g', label='test_clean_losses')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'losses.png'))
            plt.close()

            plt.plot(overall_accs, 'm', label='overall_accs')
            if args.imbalance_ratio < 1:
                plt.plot(many_accs, 'r', label='many_accs')
                plt.plot(median_accs, 'g', label='median_accs')
                plt.plot(low_accs, 'b', label='low_accs')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_accs.png'))
            plt.close()

            plt.plot(f1s, 'm', label='f1s')
            plt.grid()
            plt.legend()
            plt.savefig(os.path.join(save_dir, 'test_f1s.png'))
            plt.close()

            # save best model:
            if overall_accs[-1] > best_overall_acc:
                best_overall_acc = overall_accs[-1]
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_clean_acc.pth'))


            # save pth:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch, 
                'best_overall_acc': best_overall_acc,
                'training_losses': training_losses, 
                'test_clean_losses': test_clean_losses, 
                'f1s': f1s, 
                'overall_accs': overall_accs, 
                'many_accs': many_accs, 
                'median_accs': median_accs, 
                'low_accs': low_accs, 
                }, 
                os.path.join(save_dir, 'latest.pth'))

    # Clean up ddp:
    if args.ddp:
        cleanup()

if __name__ == '__main__':
    # get args:
    args = get_args_parser()

    # mkdirs:
    save_dir = create_save_path()
    args.save_dir = save_dir
    
    # set CUDA:
    if args.num_nodes == 1: # When using multiple nodes, we assume all gpus on each node are available.
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 

    if args.ddp:
        ngpus_per_node = torch.cuda.device_count()
        torch.multiprocessing.spawn(train, args=(ngpus_per_node,args), nprocs=ngpus_per_node, join=True)
    else:
        train(0, 0, args)
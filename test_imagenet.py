import os, argparse, time
from contextlib import ExitStack
from functools import partial
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, SVHN

from datasets.ImbalanceImageNet import LT_Dataset
from models.resnet_imagenet import ResNet50

from utils.utils import *
from utils.ltr_metrics import *
from utils.ood_metrics import *

from test import get_measures

# to prevent PIL error from reading large images:
# See https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/162#issuecomment-491115265
# or https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

def val_imagenet_only_id():
    '''
    Only evaluate ID acc.
    '''
    model.eval()
    ts = time.time()
    test_acc_meter = AverageMeter()
    labels_list = []
    pred_list = []
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits = model(images)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)

    classwise_results_dir = os.path.join(save_dir, 'classwise_results')
    create_dir(classwise_results_dir)
    for fpr_level in [0.0001, 0.001, 0.01, 0.1]:
        mask = np.load(os.path.join(pretrained_path, 'id_sample_coverage_mask', 'id_sample_coverage_mask_at_fpr%s.npy' % fpr_level))
        overall_acc = np.mean(in_preds[mask]==in_labels[mask])
        many_acc, median_acc, low_acc, _ = shot_acc(in_preds[mask], in_labels[mask], img_num_per_cls, acc_per_cls=True)
        # classwise acc:
        acc_each_class = np.full(num_classes, np.nan)
        for i in range(num_classes):
            _pred = in_preds[in_labels==i]
            _label = in_labels[in_labels==i]
            _N = np.sum(in_labels==i)
            acc_each_class[i] = np.sum(_pred==_label) / _N
        head_acc = np.mean(acc_each_class[head_class_idx])
        tail_acc = np.mean(acc_each_class[tail_class_idx])
        acc_str = 'ACC@FPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (fpr_level, overall_acc, many_acc, median_acc, low_acc, head_acc, tail_acc)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
        # classwise acc:
        if fpr_level == 0.0001:
            acc_each_class = np.full(num_classes, np.nan)
            for i in range(num_classes):
                _pred = in_preds[in_labels==i]
                _label = in_labels[in_labels==i]
                _N = np.sum(in_labels==i)
                acc_each_class[i] = np.sum(_pred==_label) / _N
            np.save(os.path.join(classwise_results_dir, 'ACC_each_class.npy'), acc_each_class)
    fp.write('\n')
    for tpr_level in [0.98, 0.95, 0.90, 0.80]:
        mask = np.load(os.path.join(pretrained_path, 'id_sample_coverage_mask', 'id_sample_coverage_mask_at_tpr%s_%s.npy' % (tpr_level, args.dout)))
        overall_acc = np.mean(in_preds[mask]==in_labels[mask])
        many_acc, median_acc, low_acc, _ = shot_acc(in_preds[mask], in_labels[mask], img_num_per_cls, acc_per_cls=True)
        acc_str = 'ACC@TPR%s: %.4f (%.4f, %.4f, %.4f)\n' % (tpr_level, overall_acc, many_acc, median_acc, low_acc)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
    fp.write('\n')
    fp.close()


def val_imagenet():
    '''
    Evaluate ID acc and OOD detection on ImageNet
    '''
    model.eval()
    ts = time.time()
    test_acc_meter = AverageMeter()
    score_list = []
    labels_list = []
    pred_list = []
    probs_list = []
    with ExitStack() as stack:
        if args.metric not in ['odin', 'maha']:
            stack.enter_context(torch.no_grad())

        for images, targets in test_loader:
            images, targets = images.cuda(), targets.cuda()
            logits, scores = get_scores_fn(model, images)
            probs = F.softmax(logits, dim=1)
            pred = logits.data.max(1)[1]
            acc = pred.eq(targets.data).float().mean()
            # append loss:
            score_list.append(scores.detach().cpu().numpy())
            labels_list.append(targets.detach().cpu().numpy())
            pred_list.append(pred.detach().cpu().numpy())
            probs_list.append(probs.max(dim=1).values.detach().cpu().numpy())
            test_acc_meter.append(acc.item())
    print('clean test time: %.2fs' % (time.time()-ts))
    # test loss and acc of this epoch:
    test_acc = test_acc_meter.avg
    in_scores = np.concatenate(score_list, axis=0)
    in_labels = np.concatenate(labels_list, axis=0)
    in_preds = np.concatenate(pred_list, axis=0)
    # in_probs = np.concatenate(probs_list, axis=0)
    if args.dout == 'imagenet-10k':
        np.save(os.path.join(save_dir, 'in_scores.npy'), in_scores)
        np.save(os.path.join(args.ckpt_path, 'in_labels.npy'), in_labels)
        np.save(os.path.join(save_dir, 'in_preds.npy'), in_preds)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

    # confidence distribution of correct samples:
    ood_score_list = []
    with ExitStack() as stack:
        if args.metric not in ['odin', 'maha']:
            stack.enter_context(torch.no_grad())

        for images, _ in ood_loader:
            images = images.cuda()
            logits, scores = get_scores_fn(model, images)
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
    ood_scores = np.concatenate(ood_score_list, axis=0)
    if args.dout == 'imagenet-10k':
        np.save(os.path.join(save_dir, 'ood_scores.npy'), ood_scores)

    auroc, aupr, fpr95 = get_measures(ood_scores, in_scores)

    # print:
    ood_detectoin_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
    print(ood_detectoin_str)
    fp.write(ood_detectoin_str + '\n')
    fp.flush()

    # ROC curve:
    print(ood_scores.shape)
    print(in_scores.shape)
    all_scores = np.concatenate([ood_scores, in_scores], axis=0)
    all_ood_labels = np.concatenate([np.ones((ood_scores.shape[0],1)), np.zeros((in_scores.shape[0],1))], axis=0)
    fpr, tpr, thresholds = roc_curve(all_ood_labels.ravel(), all_scores.ravel())

    if args.dout == 'imagenet-10k':
        # ROC curve:
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % auroc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.axhline(y=0.95, color="red", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'ROC_%s.png' % args.metric))

        # probs distributions per class:
        save_fig_dir = os.path.join(save_dir, '%s_dist' % args.metric)
        create_dir(save_fig_dir)
        in_correct = (in_labels==in_preds)
        for c in np.unique(in_labels):
            scores_c = in_scores[in_labels==c]
            correct_c = in_correct[in_labels==c]
            plt.figure()
            if args.metric in ['energy']:
                plt.hist(scores_c[correct_c], bins=20, color='g')
            else:
                plt.hist(-scores_c[correct_c], bins=20, color='g')
            plt.savefig(os.path.join(save_fig_dir, '%d_correct.png' % c))
            plt.close()
            plt.figure()
            if args.metric in ['energy']:
                plt.hist(scores_c[~correct_c], bins=20, color='r')
            else:
                plt.hist(-scores_c[~correct_c], bins=20, color='r')
            plt.savefig(os.path.join(save_fig_dir, '%d_wrong.png' % c))
            plt.close()
        plt.figure()
        # plt.scatter(np.ones_like(ood_probs), ood_probs, marker='*', c='b')
        if args.metric in ['energy']:
            plt.hist(ood_scores, bins=100)
        else:
            plt.hist(-ood_scores, bins=100)
        plt.savefig(os.path.join(save_fig_dir, 'ood.png'))
        plt.close()

    save_mask_dir = os.path.join(save_dir, 'id_sample_coverage_mask')
    create_dir(save_mask_dir)
    classwise_results_dir = os.path.join(save_dir, 'classwise_results')
    create_dir(classwise_results_dir)
    # classwise acc:
    acc_each_class = np.full(num_classes, np.nan)
    for i in range(num_classes):
        _pred = in_preds[in_labels==i]
        _label = in_labels[in_labels==i]
        _N = np.sum(in_labels==i)
        acc_each_class[i] = np.sum(_pred==_label) / _N
    np.save(os.path.join(classwise_results_dir, 'ACC_each_class.npy'), acc_each_class)
    # Find ACC@FPRn:
    for fpr_level in [0.0001, 0.001, 0.01, 0.1]:
        cutoff = np.argmin(np.abs(fpr - fpr_level))
        score_threshold = thresholds[cutoff]
        print('fpr_level:', fpr_level)
        print('score_threshold:', score_threshold)
        print('fpr:', fpr[cutoff])
        print('tpr:', tpr[cutoff])
        in_preds_above_score_threshold = in_preds[in_scores<=score_threshold] # NOTE: score is MINUS MSP
        in_labels_above_score_threshold = in_labels[in_scores<=score_threshold]
        acc_at_fpr_level = np.mean(in_preds_above_score_threshold==in_labels_above_score_threshold)
        many_acc, median_acc, low_acc, _ = shot_acc(in_preds_above_score_threshold, in_labels_above_score_threshold, img_num_per_cls, acc_per_cls=True)
        # classwise acc:
        acc_each_class = np.full(num_classes, np.nan)
        for i in range(num_classes):
            _pred = in_preds[in_labels==i]
            _label = in_labels[in_labels==i]
            _N = np.sum(in_labels==i)
            acc_each_class[i] = np.sum(_pred==_label) / _N
        head_acc = np.mean(acc_each_class[head_class_idx])
        tail_acc = np.mean(acc_each_class[tail_class_idx])
        acc_str = 'ACC@FPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (
            fpr_level, acc_at_fpr_level, many_acc, median_acc, low_acc, head_acc, tail_acc)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
        # save mask:
        mask = in_scores<=score_threshold
        np.save(os.path.join(save_mask_dir, 'id_sample_coverage_mask_at_fpr%s.npy' % (fpr_level)), mask)
    fp.write('\n')

    # Find ACC@TPRn:
    for tpr_level in [0.98, 0.95, 0.90, 0.80]:
        cutoff = np.argmin(np.abs(tpr - tpr_level))
        score_threshold = thresholds[cutoff]
        print('tpr_level:', tpr_level)
        print('score_threshold:', score_threshold)
        print('fpr:', fpr[cutoff])
        print('tpr:', tpr[cutoff])
        in_preds_above_score_threshold = in_preds[in_scores<=score_threshold]
        in_labels_above_score_threshold = in_labels[in_scores<=score_threshold]
        acc_at_tpr_level = np.mean(in_preds_above_score_threshold==in_labels_above_score_threshold)
        many_acc, median_acc, low_acc, _ = shot_acc(in_preds_above_score_threshold, in_labels_above_score_threshold, img_num_per_cls, acc_per_cls=True)
        acc_str = 'ACC@TPR%s: %.4f (%.4f, %.4f, %.4f)\n' % (tpr_level, acc_at_tpr_level, many_acc, median_acc, low_acc)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
        # save mask:
        mask = in_scores<=score_threshold
        np.save(os.path.join(save_mask_dir, 'id_sample_coverage_mask_at_tpr%s_%s.npy' % (tpr_level, args.dout)), mask)
    fp.write('\n')

    # Find FPR@TPRn:
    for tpr_level in [0.98, 0.95, 0.90, 0.80]:
        cutoff = np.argmin(np.abs(tpr - tpr_level))
        score_threshold = thresholds[cutoff]
        print('tpr_level:', tpr_level)
        print('score_threshold:', score_threshold)
        print('fpr:', fpr[cutoff])
        print('tpr:', tpr[cutoff])
        in_labels_above_score_threshold = in_labels[in_scores<=score_threshold]
        rejection_ratio = []
        for i in range(num_classes):
            rejection_ratio_i = 1 - np.sum(in_labels_above_score_threshold==i) / np.sum(in_labels==i)
            rejection_ratio.append(rejection_ratio_i)
        many_rejection_ratio = np.mean(rejection_ratio[0:int(0.34*num_classes)])
        median_rejection_ratio = np.mean(rejection_ratio[int(0.34*num_classes):int(0.67*num_classes)+1])
        low_rejection_ratio = np.mean(rejection_ratio[int(0.67*num_classes)+1:])
        
        rejection_ratio = np.array(rejection_ratio)
        head_rejection_ratio = np.mean(rejection_ratio[head_class_idx])
        tail_rejection_ratio = np.mean(rejection_ratio[tail_class_idx])
            
        acc_str = 'FPR@TPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (tpr_level, fpr[cutoff], 
            many_rejection_ratio, median_rejection_ratio, low_rejection_ratio, head_rejection_ratio, tail_rejection_ratio)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
        # classwise FPR:
        np.save(os.path.join(classwise_results_dir, 'FPR%d_each_class_%s.npy' % (int(tpr_level*100), args.dout)), rejection_ratio)
    fp.write('\n')

    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test an ImageNet Classifier')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--num_workers', type=int, default=32)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='imagenet', choices=['imagenet'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets/', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='imagenet10k', choices=['imagenet10k'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet50', choices=['ResNet50'], help='which model to use')
    # 
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
    parser.add_argument('--metric', default='msp', choices=['msp'], help='OOD detection metric')
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--ckpt', default='latest', choices=['latest', 'best'])
    parser.add_argument('--only_id', action='store_true', help='If true, only test ID acc')
    parser.add_argument('--tnorm', action='store_true', help='If true, use t-norm for LT inference')
    args = parser.parse_args()
    print(args)

    # load id data selection mask:
    if args.only_id:
        pretrained_path = os.path.join(args.ckpt_path, os.pardir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.tnorm:
        save_dir = os.path.join(args.ckpt_path, 'tnorm')
    else:
        save_dir = os.path.join(args.ckpt_path)
    create_dir(save_dir)

    # data:
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
    num_classes = 1000
    train_set = LT_Dataset(
        os.path.join(args.data_root_path, 'imagenet'), './datasets/ImageNet_LT/ImageNet_LT_train.txt', transform=train_transform, 
        subset_class_idx=np.arange(0,1000))
    test_set = ImageFolder(os.path.join(args.data_root_path, 'imagenet', 'val'), transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    din_str = 'Din is %s with %d images' % (args.dataset, len(test_set))
    print(din_str)
    
    ood_set = ImageFolder(os.path.join(args.data_root_path, 'imagenet_ood_test_1k'), transform=test_transform)
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    dout_str = 'Dout is %s with %d images' % (args.dout, len(ood_set))
    print(dout_str)

    img_num_per_cls = np.array(train_set.img_num_per_cls)

    idx = np.argsort(img_num_per_cls)
    tail_class_idx = idx[:int(num_classes*0.5)]
    head_class_idx = idx[int(num_classes*0.5):]

    # model:
    model = ResNet50(num_classes=num_classes).cuda()
    model = torch.nn.DataParallel(model)

    # load model:
    if args.ckpt == 'latest':
        ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'))['model']
    else:
        ckpt = torch.load(os.path.join(args.ckpt_path, 'best_clean_acc.pth'))
    model.load_state_dict(ckpt, strict=False)   
    model.requires_grad_(False)

    # select a detection function:
    get_scores_fn = get_msp_scores

    # log file:
    if args.tnorm:
        '''
        Decoupling representation and classifier for long-tailed recognition. ICLR, 2020.
        '''
        w = model.linear.weight.data
        w_row_norm = torch.norm(w, p='fro', dim=1)
        print(w_row_norm)
        model.linear.weight.data = w / w_row_norm[:,None]
        model.linear.bias.zero_()
    test_result_file_name = 'test_results_%s.txt' % (args.metric)
    fp = open(os.path.join(save_dir, test_result_file_name), 'a+')
    fp.write('\n===%s===\n' % (args.dout))
    fp.write(din_str + '\n')
    fp.write(dout_str + '\n')

    if args.only_id:
        val_imagenet_only_id()
    else:
        val_imagenet()
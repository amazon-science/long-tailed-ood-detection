'''
Codes adapted from https://github.com/hendrycks/outlier-exposure/blob/master/CIFAR/test.py
which uses Apache-2.0 license.
'''
import os, argparse, time
from contextlib import ExitStack
from functools import partial
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.SCOODBenchmarkDataset import SCOODDataset
from models.resnet import ResNet18, ResNet34

from utils.utils import *
from utils.ltr_metrics import *
from utils.ood_metrics import *

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

## Test on CIFAR:   
def val_cifar_only_id():
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
        head_acc = np.mean(acc_each_class[0:int(0.5*num_classes)])
        tail_acc = np.mean(acc_each_class[int(0.5*num_classes):int(num_classes)])
        acc_str = 'ACC@FPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (fpr_level, overall_acc, many_acc, median_acc, low_acc, head_acc, tail_acc)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
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


def val_cifar():
    '''
    Evaluate ID acc and OOD detection on CIFAR10/100
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
    if args.dout == 'svhn':
        np.save(os.path.join(save_dir, 'in_scores.npy'), in_scores)
        np.save(os.path.join(args.ckpt_path, 'in_labels.npy'), in_labels)
        np.save(os.path.join(save_dir, 'in_preds.npy'), in_preds)
    many_acc, median_acc, low_acc, _ = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc)
    print(clean_str)
    fp.write(clean_str + '\n')
    fp.flush()

    # confidence distribution of correct samples:
    ood_score_list, sc_labels_list = [], []
    with ExitStack() as stack:
        if args.metric not in ['odin', 'maha']:
            stack.enter_context(torch.no_grad())

        for images, sc_labels in ood_loader:
            images, sc_labels = images.cuda(), sc_labels.cuda()
            logits, scores = get_scores_fn(model, images)
            # append loss:
            ood_score_list.append(scores.detach().cpu().numpy())
            sc_labels_list.append(sc_labels.detach().cpu().numpy())
    ood_scores = np.concatenate(ood_score_list, axis=0)
    sc_labels = np.concatenate(sc_labels_list, axis=0)
    if args.dout == 'svhn':
        np.save(os.path.join(save_dir, 'ood_scores.npy'), ood_scores)

    # move some elements in ood_scores to in_scores:
    print('in_scores:', in_scores.shape)
    print('ood_scores:', ood_scores.shape)
    fake_ood_scores = ood_scores[sc_labels>=0]
    real_ood_scores = ood_scores[sc_labels<0]
    real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)
    print('fake_ood_scores:', fake_ood_scores.shape)
    print('real_in_scores:', real_in_scores.shape)
    print('real_ood_scores:', real_ood_scores.shape)

    auroc, aupr, fpr95 = get_measures(real_ood_scores, real_in_scores)

    # print:
    ood_detectoin_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
    print(ood_detectoin_str)
    fp.write(ood_detectoin_str + '\n')
    fp.flush()

    # ROC curve:
    print(ood_scores.shape)
    print(in_scores.shape)
    all_scores = np.concatenate([real_ood_scores, real_in_scores], axis=0)
    all_ood_labels = np.concatenate([np.ones((real_ood_scores.shape[0],1)), np.zeros((real_in_scores.shape[0],1))], axis=0)
    fpr, tpr, thresholds = roc_curve(all_ood_labels.ravel(), all_scores.ravel())

    if args.dout == 'svhn':
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
        head_acc = np.mean(acc_each_class[0:int(0.5*num_classes)])
        tail_acc = np.mean(acc_each_class[int(0.5*num_classes):int(num_classes)])
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

        head_rejection_ratio = np.mean(rejection_ratio[0:int(0.5*num_classes)])
        tail_rejection_ratio = np.mean(rejection_ratio[int(0.5*num_classes):int(num_classes)])
            
        acc_str = 'FPR@TPR%s: %.4f (%.4f, %.4f, %.4f | %.4f, %.4f)\n' % (tpr_level, fpr[cutoff], 
            many_rejection_ratio, median_rejection_ratio, low_rejection_ratio, head_rejection_ratio, tail_rejection_ratio)
        print(acc_str)
        fp.write(acc_str)
        fp.flush()
    fp.write('\n')

    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
    parser.add_argument('--gpu', default='3')
    parser.add_argument('--num_workers', type=int, default=4)
    # dataset:
    parser.add_argument('--dataset', '--ds', default='cifar10', choices=['cifar10', 'cifar100'], help='which dataset to use')
    parser.add_argument('--data_root_path', '--drp', default='/ssd1/haotao/datasets/', help='Where you save all your datasets.')
    parser.add_argument('--dout', default='svhn', choices=['svhn', 'places365', 'cifar', 'texture', 'tin', 'lsun'], help='which dout to use')
    parser.add_argument('--model', '--md', default='ResNet18', choices=['ResNet18', 'ResNet34', 'WRN40'], help='which model to use')
    # 
    parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
    parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
    parser.add_argument('--metric', default='msp', choices=['msp'], help='OOD detection metric')
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument('--ckpt', default='latest', choices=['latest', 'best'])
    parser.add_argument('--only_id', action='store_true', help='If true, only test ID acc')
    args = parser.parse_args()
    print(args)

    # load id data selection mask:
    if args.only_id:
        pretrained_path = os.path.join(args.ckpt_path, os.pardir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    save_dir = os.path.join(args.ckpt_path)
    create_dir(save_dir)

    # data:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])
    if args.dataset == 'cifar10':
        num_classes = 10
        train_set = IMBALANCECIFAR10(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = IMBALANCECIFAR100(train=True, transform=train_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path)
        test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, 
                                drop_last=False, pin_memory=True)
    if args.dout == 'cifar':
        if args.dataset == 'cifar10':
            args.dout = 'cifar100'
        elif args.dataset == 'cifar100':
            args.dout = 'cifar10'
    ood_set = SCOODDataset(os.path.join(args.data_root_path, 'SCOOD'), id_name=args.dataset, ood_name=args.dout, transform=test_transform)
    ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    print('Dout is %s with %d images' % (args.dout, len(ood_set)))

    img_num_per_cls = np.array(train_set.img_num_per_cls)

    # model:
    if args.model == 'ResNet18':
        model = ResNet18(num_classes=num_classes).cuda()
    elif args.model == 'ResNet34':
        model = ResNet34(num_classes=num_classes).cuda()
    # model = torch.nn.DataParallel(model)

    # load model:
    if args.ckpt == 'latest':
        ckpt = torch.load(os.path.join(args.ckpt_path, 'latest.pth'))['model']
    else:
        ckpt = torch.load(os.path.join(args.ckpt_path, 'best_clean_acc.pth'))
    model.load_state_dict(ckpt, strict=False)   
    model.requires_grad_(False)

    # select a detection function:
    if args.metric == 'msp':
        get_scores_fn = get_msp_scores

    # log file:
    test_result_file_name = 'test_results_%s.txt' % (args.metric)
    fp = open(os.path.join(save_dir, test_result_file_name), 'a+')
    fp.write('\n===%s===\n' % (args.dout))

    if args.only_id:
        val_cifar_only_id()
    else:
        val_cifar()

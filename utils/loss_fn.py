'''
my_cl_loss_fn2 function is adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py
which is originally licensed under BSD-2-Clause.
'''

import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.loss import SoftTargetCrossEntropy

def prior_to_tau(prior, tau0=0.1):
    '''
    Args:
        prior: iterable with len=num_classes
    
    Returns:
        tau: iterable with len=num_classes
    '''
    tau = tau0 / (prior[0]-prior[-1]) * (prior-prior[-1])
    return tau

def my_cl_loss_fn3(f_id, f_ood, labels, temperature=0.07, ls=False, tau_list=None, reweighting=False, w_list=None):
    '''
    A variant of supervised contrastive loss: 
    push ID samples from ID samples of different classes;
    push ID samples from OOD samples, but using different push strength according to prior distribution P(y);
    pull ID samples within the same classes.

    Args:
        f_id: features of ID_tail samples. Tensor. Shape=(N_id+N_ood,N_view,d)
        f_ood: features of OE samples. Tensor. Shape=(N_ood,d)
        labels: labels of ID_tail samples.
        ls: Bool. True if do label smoothing on CL loss labels.
        tau_list: list of floats. len=num_classes. Label smoothing parameter for each class based on prior p(y).
    '''

    f_id = f_id.view(f_id.shape[0], f_id.shape[1], -1) # shape=(N_id,2,d), i.e., 2 views

    N_id = f_id.shape[0]
    N_ood = f_ood.shape[0]
    labels = labels.contiguous().view(-1, 1)
    
    N_views = f_id.shape[1] # = 2
    f_id = torch.cat(torch.unbind(f_id, dim=1), dim=0) # shape=(N_id*2,d)

    # compute logits
    anchor_dot_contrast = torch.div(
        torch.matmul(f_id, torch.cat((f_id, f_ood), dim=0).T),
        temperature) # shape=(2N_id,2*N_id+N_ood)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # dim=1 is the KL dim.
    logits = anchor_dot_contrast - logits_max.detach() # shape=(2N_id,2*N_id+N_ood)
    logits = logits.masked_select(~torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)).view(logits.shape[0], logits.shape[1]-1)  # remove self-contrast cases (diag elements)
    
    # labels for CL:
    mask = torch.eq(labels, labels.T).float().to(labels.device) # shape=(N_id,N_id). 1 -> positive pair
    mask = mask.repeat(N_views, N_views) # shape=(2*N_id,2*N_id)
    mask = torch.cat((mask, torch.zeros(mask.shape[0],N_ood).to(mask.device)),dim=1) # all ood samples are negative samples to ID samples. shape=(2*N_id,2*N_id+N_ood)
    mask = mask.masked_select(~torch.eye(mask.shape[0], mask.shape[1], dtype=bool).to(mask.device)).view(mask.shape[0], mask.shape[1]-1) # remove self-contrast cases (diag elements). shape=(2*N_id,2*N_id-1+N_ood)
    cl_labels = nn.functional.normalize(mask, dim=1, p=1) # so that each row has sum 1. shape=(2*N_id,2*N_id-1+N_ood)

    # label smoothing:
    if ls:
        for _c, tau in enumerate(tau_list):
            _c_idx = labels == _c
            _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()
            cl_labels[_c_idx] *= 1 - tau
            cl_labels[_c_idx,2*N_id:] = tau / N_ood

    # loss
    loss = torch.sum(-cl_labels * F.log_softmax(logits, dim=-1), dim=-1)

    # reweighting:
    if reweighting:
        assert ls is False
        for _c, w in enumerate(w_list):
            _c_idx = labels == _c
            if torch.sum(_c_idx) > 0:
                assert w > 0, ("Negative loss weight value detected: %s among %s when c=%s among %s" % (w, w_list, _c, torch.unique(labels)))
                _c_idx = torch.cat([_c_idx,_c_idx], dim=0).squeeze()
                loss[_c_idx] *= w
    
    # mean over the batch:
    loss = loss.mean() # average among all rows

    return loss

def oe_loss_fn(logits):
    '''
    The original instable implementation. torch.logsumexp is not numerically stable.
    '''
    return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()


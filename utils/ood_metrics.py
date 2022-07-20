import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np 
import sklearn.covariance
from advertorch.utils import clamp


def get_msp_scores(model, images):

    logits = model(images)
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values

    scores = - msp # The larger MSP, the smaller uncertainty

    return logits, scores



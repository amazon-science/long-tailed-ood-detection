import os, ast
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

class SCOODDataset(torch.utils.data.Dataset):

    def __init__(self, root, id_name, ood_name, transform):

        super(SCOODDataset, self).__init__()

        assert id_name in ['cifar10', 'cifar100']

        imglist_path = os.path.join(root, 'data/imglist/benchmark_%s' % id_name, 'test_%s.txt' % ood_name)

        with open(imglist_path) as fp:
            self.imglist = fp.readlines()
        
        self.transform = transform
        self.root = root

        print("SCOODDataset (id %s, ood %s) Contain %d images" % (id_name, ood_name, len(self.imglist)))

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # parse the string in imglist file:
        line = self.imglist[index].strip("\n")
        tokens = line.split(" ", 1)
        image_name, extra_str = tokens[0], tokens[1]
        extras = ast.literal_eval(extra_str)
        sc_label = extras['sc_label'] # the ood label is here. -1 means ood.

        # read image according to image name:
        img_path = os.path.join(self.root, 'data', 'images', image_name)
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, sc_label

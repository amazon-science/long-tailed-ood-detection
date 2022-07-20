'''From https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch/blob/master/classification/data/dataloader.py'''
from torch.utils.data import Dataset
import os
import numpy as np 
from PIL import Image

class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, subset_class_idx, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                _label = int(line.split()[1])
                if _label in subset_class_idx:
                    self.labels.append(_label)
                    self.img_path.append(os.path.join(root, line.split()[0]))
               
        num_classes = len(subset_class_idx)
        self.img_num_per_cls = np.zeros(num_classes)
        for i in range(num_classes):
            self.img_num_per_cls[i] = np.sum(np.array(self.labels)==subset_class_idx[i])
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

class TinyImages(torch.utils.data.Dataset):

    def __init__(self, root, transform):

        super(TinyImages, self).__init__()

        self.data = np.load(os.path.join(root, 'tinyimages80m', '300K_random_images.npy'))
        self.transform = transform

        print("TinyImages Contain {} images".format(len(self.data)))

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, -1  # -1 is the class

    def __len__(self):
        return len(self.data)


def tinyimages300k_dataloaders(num_samples=300000, train_batch_size=64, num_workers=8, data_root_path='/ssd1/haotao/datasets'):

    num_samples = int(num_samples)

    data_dir = os.path.join(data_root_path)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_set = Subset(TinyImages(data_dir, train=True, transform=train_transform, download=True), list(range(num_samples)))

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers,
                                drop_last=True, pin_memory=True)

    return train_loader
    
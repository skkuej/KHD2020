import glob, os, re, random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils import cutout, cutout2


class ImageDataset(Dataset):
    def __init__(self, mode, transform=None, augmentation=0, smaller=0, filenames=None):
        super(ImageDataset, self).__init__()
        self.mode = mode

        if self.mode == 'train':
            self.root = '/DATA/data_cancer/train'
        elif self.mode == 'test':
            self.root = '/DATA/data_cancer/test'

        self.transform = transform
        self.augmentation = augmentation
        self.smaller = smaller
        if filenames:
            self.filenames = filenames
        else:
            self.filenames = glob.glob(os.path.join(self.root,'*.jpg'))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        img, label, target, location = read_data(self.filenames[int(index)], self.smaller)
        if self.augmentation:
            if self.augmentation==2 and random.random()>0.75:
                img = cutout(img)
            if (self.augmentation==3 or self.augmentation==5) and random.random()>0.75:
                img = cutout2(img)

            img = Image.fromarray(img)
            '''
            if self.augmentation==3 and random.random()>0.8:
                self.transform = transforms.Compose([self.transform, transforms.GaussianBlur(0.3)])
            '''
            img = self.transform(img)
        else:
            img = self.transform(img)

        if self.mode == 'train':
            sample = {'image': img, 'label': label, 'target': target, 'location': location}
        elif self.mode == 'test':
            remove_path = lambda x : re.sub(r'^.+/', '', x)
            code = remove_path(self.filenames[int(index)])
            sample = {'image': img, 'code': code}
        return sample

    def get_labels(self):
        targets = [re.sub(r'^.+/','',x).replace('.jpg','').split('_')[-1] for _,x in enumerate(self.filenames)]
        labels = []
        for x in targets:
            if x == 'C': labels.append(1)
            else: labels.append(0)
        return np.array(labels)


def read_data(fn, smaller):
    label = None
    target = None

    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)  # (1115, 1668, 3)
    if smaller:
        img = cv2.resize(img, dsize=(456, 456))
    else:
        img = cv2.resize(img, dsize=(512, 512))
    small_size = 1
    splitUnderbar = lambda x : re.sub(r'^.+/','',x).replace('.jpg','').split('_')
    target = splitUnderbar(fn)[-1]
    location = splitUnderbar(fn)[-2]
    if target == 'C':
        label = 1
    elif target == 'B' or target == 'N':
        label = 0

    if location == 'T': location=0
    elif location == 'P': location=1
    elif location == 'B': location=2
    elif location == 'O': location=3
    return img, label, target, location


def test_example():

    custom_transformer = transforms.Compose([
                transforms.ToTensor(),
                ])

    train_dataset = ImageDataset(mode='train', transform=custom_transformer)
    test_dataset = ImageDataset(mode='test', transform=custom_transformer)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(len(train_dataset)) # 4457
    print(len(testloader)) # 1117

    img, target = next(iter(trainloader))
    print(img.shape)
    # return: torch.Size([1, 3, 1080, 1920])
    print(target)
    # return: ('B',)

    img, target = next(iter(testloader))
    print(img.shape)
    # return: torch.Size([1, 3, 424, 616])
    print(target)
    # return: ('X',)


if __name__ == "__main__":
    test_example()

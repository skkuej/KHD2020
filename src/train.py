import os, sys, glob, re
import random
import numpy as np
import time
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2

basepath = '/USER/INFERENCE/CANCER/'
sys.path.append(basepath + 'src/')
from utils import seed_everything, divide_fold, my_model, get_transform
from dataloader_cancer import ImageDataset
from classifier import train,test


#trial_no = 1
seed=20
seed_everything(seed)

num_channels = 3
num_classes = 2
num_fold = 5

def main(N_fold, batch_size, lr, n_epochs, num_workers, split_ratio, model_type, weight_filename, trial_no, augmentation, small_size, shuffle_data, class_balance, skip_batch, prefix):
    if not os.path.exists(basepath+'val_perf'):
        os.makedirs(basepath+'val_perf')

    #N_fold = 4

    TRANSFORM = get_transform(augmentation)

    '''
    Dataset
    '''
    filenames = glob.glob('/DATA/data_cancer/train/*.jpg')
    targets = [re.sub(r'^.+/','',x).replace('.jpg','').split('_')[-1] for _,x in enumerate(filenames)]
    labels = []
    for x in targets:
        if x == 'C': labels.append(1)
        else: labels.append(0)

    num_train = len(labels)
    train_indices = divide_fold(np.array(labels), num_fold)[N_fold]
    if shuffle_data:
       np.random.shuffle(train_indices)
    print('** train_indices: ', train_indices)
    val_indices = np.setdiff1d(range(num_train), train_indices)
    del targets, labels

    filenames = np.array(filenames)
    print('** do_augment: ', augmentation)
    if augmentation:
        stack_train = []
        num_train = len(train_indices)
        #for aug in aug_transform.keys():
        stack_train.append(ImageDataset('train', TRANSFORM['base'], augmentation, small_size, filenames[train_indices].tolist()))
        if augmentation < 3:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/2), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['affine'], augmentation, small_size, fn.tolist()))
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/2), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['flip'], augmentation, small_size, fn.tolist()))
        elif augmentation == 3:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/4), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['blur'], augmentation, small_size, fn.tolist()))
import os, sys, glob, re
import random
import numpy as np
import time
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2

basepath = '/USER/INFERENCE/CANCER/'
sys.path.append(basepath + 'src/')
from utils import seed_everything, divide_fold, my_model, get_transform
from dataloader_cancer import ImageDataset
from classifier import train,test


#trial_no = 1
seed=20
seed_everything(seed)

num_channels = 3
num_classes = 2
num_fold = 5

def main(N_fold, batch_size, lr, n_epochs, num_workers, split_ratio, model_type, weight_filename, trial_no, augmentation, small_size, shuffle_data, class_balance, skip_batch, prefix):
    if not os.path.exists(basepath+'val_perf'):
        os.makedirs(basepath+'val_perf')

    #N_fold = 4

    TRANSFORM = get_transform(augmentation)

    '''
    Dataset
    '''
    filenames = glob.glob('/DATA/data_cancer/train/*.jpg')
    targets = [re.sub(r'^.+/','',x).replace('.jpg','').split('_')[-1] for _,x in enumerate(filenames)]
    labels = []
    for x in targets:
        if x == 'C': labels.append(1)
        else: labels.append(0)

    num_train = len(labels)
    train_indices = divide_fold(np.array(labels), num_fold)[N_fold]
    if shuffle_data:
       np.random.shuffle(train_indices)
    print('** train_indices: ', train_indices)
    val_indices = np.setdiff1d(range(num_train), train_indices)
    del targets, labels

    filenames = np.array(filenames)
    print('** do_augment: ', augmentation)
    if augmentation:
        stack_train = []
        num_train = len(train_indices)
        #for aug in aug_transform.keys():
        stack_train.append(ImageDataset('train', TRANSFORM['base'], augmentation, small_size, filenames[train_indices].tolist()))
        if augmentation < 3:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/2), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['affine'], augmentation, small_size, fn.tolist()))
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/2), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['flip'], augmentation, small_size, fn.tolist()))
        elif augmentation == 3:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/4), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['blur'], augmentation, small_size, fn.tolist()))
        elif augmentation == 4:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/5), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['affine'], augmentation, small_size, fn.tolist()))
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/5), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['flip'], augmentation, small_size, fn.tolist()))
        elif augmentation == 5:
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/5), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['rotation'], augmentation, small_size, fn.tolist()))
            for i in range(augmentation):
                fn = filenames[np.random.choice(num_train, int(num_train/5), replace=False)]
            stack_train.append(ImageDataset('train', TRANSFORM['blur'], augmentation, small_size, fn.tolist()))

        train_set = torch.utils.data.ConcatDataset(stack_train)
        del stack_train
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        val_set = ImageDataset('train', TRANSFORM['base'], augmentation, small_size, filenames[val_indices].tolist())
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
        del train_set, val_set, filenames
    else:
        train_set = ImageDataset('train', TRANSFORM['base'], augmentation, small_size)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices), num_workers=num_workers)
        valloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=SubsetRandomSampler(val_indices), num_workers=num_workers)
        del train_set, filenames

    '''
    Model parameters
    '''
    print(f'num_train : {len(trainloader)}, num_val : {len(valloader)}')

    model = my_model(num_channels, num_classes, model_type)
    print('model_filename: ', weight_filename)
    if weight_filename:
        model.load_state_dict(torch.load(f'{basepath}weights/{weight_filename}.pth'))
        seperateInt = re.findall('\d+', weight_filename)
        trial_no = int(seperateInt[-2])
        base_epoch = int(seperateInt[-1]) + 1
    else:
        base_epoch = 0

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    if class_balance:
        '''
        sample size-dependent class weight
        '''
        #labels = train_set.get_labels()
        #nsamples = [len(np.where(labels==i)[0]) for i in range(num_classes)]
        #print('train.py - The number of classes in data: ', nsamples)
        #class_weight = torch.FloatTensor([1-(x/sum(nsamples)) for x in nsamples]).to(device)
        class_weight = torch.FloatTensor([1.0, 1.8]).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight, reduction='mean')
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    print(f'n_epoch:{n_epochs}, lr:{lr}, batch_size:{batch_size}')

    best_model = train(model, n_epochs, trainloader, valloader, criterion, optimizer, None, device, trial_no, base_epoch, prefix, skip_batch)
    #np.save(f'{basepath}val_auc/trial_{trial_no}.npy', val_auc)
    #np.save(f'{basepath}val_auc/lr_trial_{trial_no}.npy', lr_step)

    print("done")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Step1_ Train classifier")
    parser.add_argument('-b', dest='N_batch', default=16, type=int, required=False, help='Batch size')
    parser.add_argument('-l', dest='lr', default=0.00000001, type=float, required=False, help='Learning rate')
    parser.add_argument('-e', dest='N_epoch', default=5, type=int, required=False, help='Number of epoch')
    parser.add_argument('-w', dest='N_worker', default=16, type=int, required=False, help='Number of worker')
    parser.add_argument('-r', dest='split_ratio', default=0.8, type=float, required=False, help='Validation split ratio')
    parser.add_argument('-m', dest='model_type', default='efficientnet-b0', type=str, required=False, help='Type of pretrained model')
    parser.add_argument('-f', dest='weight_filename', default=None, type=str, required=False, help='Filename for model weight')
    parser.add_argument('-t', dest='N_trial', default=0, type=int, required=False, help='Trial number')
    parser.add_argument('--nf', dest='nfold', default=1, type=int, required=False, help='Trial number')
    parser.add_argument('-a', dest='augmentation', default=0, type=int, required=False, help='Augment data or not')
    parser.add_argument('-s', dest='small_size', default=0, type=int, required=False, help='Resize data to 224')
    parser.add_argument('-d', dest='shuffle_data', default=0, type=int, required=False, help='Shuffle data or not')
    parser.add_argument('-c', dest='class_weight', default=0, type=int, required=False, help='Perform class balancing or not')
    parser.add_argument('-p', dest='prefix', default=None, type=str, required=False, help='Prefix to save result files')
    parser.add_argument('--skip', dest='skip_batch', default=0, type=int, required=False, help='Skip a few batch per epoch')

    args = parser.parse_args()

    main(args.nfold, args.N_batch, args.lr, args.N_epoch, args.N_worker, args.split_ratio, args.model_type, args.weight_filename, args.N_trial, args.augmentation, args.small_size, args.shuffle_data, args.class_weight, args.skip_batch, args.prefix)

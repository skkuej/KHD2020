import os
import random
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available():
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def label_to_num(targets):
    labels = []
    for x in targets:
        if x == 'C': labels.append(1)
        elif x == 'B': labels.append(0)
        elif x == 'N': labels.append(-1)
        else: print("ERROR: Invalid class type")
    return np.array(labels)


def train_split(labels, ratio):
    num_classes = len(np.unique(labels))
    #num_train = len(labels)

    train_indices = np.array([])
    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        num_class = len(class_indices)
        num_train = round(ratio*num_class)
        for i in range(n_shuffle + 1):
            selected = np.random.choice(num_class, num_train, replace=False)
        train_indices = np.hstack((train_indices, class_indices[selected]))

    np.random.shuffle(train_indices)
    return train_indices.astype(int)


def divide_fold(labels=None, num_fold=5):
    num_classes = len(np.unique(labels))
    num_total = len(labels)
    num_val = round(num_total/num_fold)
    num_train = num_total - num_val
    train_indices = np.zeros((num_fold, num_train))
    ''' fold 1 '''
    tidx = np.array([])
    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        num_class = len(class_indices)
        num_train = round(num_class*(1-1/num_fold))
        selected = np.random.choice(num_class, num_train, replace=False)
        tidx = np.hstack((tidx, class_indices[selected])).astype(int)
    np.random.shuffle(tidx)
    train_indices[0,:] = tidx

    ''' fold 2 - 5 '''
    labels = np.array([labels[i] for i in tidx])
    num_fold = num_fold-1
    val_indices = np.zeros((num_fold, num_val))
    ii = 0
    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        num_class = len(class_indices)
        num_train = round(num_class/num_fold)
        num_val = int(num_class/num_fold)
        np.random.shuffle(class_indices)
        class_indices = np.array([tidx[i] for i in class_indices])
        indices = np.reshape(class_indices[:num_fold*num_val], (num_fold, num_val))
        val_indices[:, ii:ii+num_val] = indices
        ii = ii + num_val

    for i in range(num_fold):
        buf = np.setdiff1d(range(num_total), val_indices[i])
        np.random.shuffle(buf)
        train_indices[i+1] = buf
    return train_indices.astype(np.int)


def divide_into_(labels, num_fold):
    num_classes = len(np.unique(labels))
    num_train = len(labels)

    val_indices = np.zeros((num_fold, round(num_train/num_fold)))
    ii = 0
    for i in range(num_classes):
        class_indices = np.where(labels == i)[0]
        num_class = len(class_indices)
        num_val = int(num_class/num_fold)
        np.random.shuffle(class_indices)
        indices = np.reshape(class_indices[:num_fold*num_val], (num_fold, num_val))

        val_indices[:, ii:ii+num_val] = indices
        ii = ii + num_val

    for i in range(num_fold):
        buf = val_indices[i]
        np.random.shuffle(buf)
        val_indices[i] = buf
    return val_indices.astype(np.int)


def my_model(num_channels, num_classes, model_type='efficientnet-b0'):
    if num_channels != 3:
        print("ERROR: The number of the channel is not 3.")

    if model_type == 'resnet18':
        model = models.resnet18(pretrained=True)
        # models.fc: 512 -> 1000
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)     # 512

    elif model_type == 'resnet50':
        model = models.resnet50(pretrained=True)
        # models.fc: 2048 -> 1000
        #model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)    # 2048
        classifier = nn.Linear(in_features=model.fc.out_features, out_features=num_classes, bias=True)  # 1000
        model = nn.Sequential(model, classifier)

    elif model_type.startswith('efficientnet'):
        if model_type == 'efficientnet-b5':
            model = EfficientNet.from_pretrained(model_type)
            # b7: models._fc: 2048 -> 1000
            fc = nn.Linear(in_features=model._fc.out_features, out_features=128, bias=True)   # 1000
            classifier = nn.Linear(in_features=128, out_features=num_classes, bias=True)   # 1000
            model = nn.Sequential(model, fc, nn.Dropout(0.5), classifier)
        else:
            model = EfficientNet.from_pretrained(model_type, num_classes=num_classes)
        # b0: models._fc: 1280 -> 1000
        # b2: models._fc: 1408 -> 1000
        # b4: models._fc: 1792 -> 1000
        # b6: models._fc: 2304 -> 1000

    elif model_type == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True)
        # models.fc: 1280 -> 1000
        classifier = nn.Linear(in_features=model.classifier[1].out_features, out_features=num_classes, bias=True)   # 1000
        model = nn.Sequential(model, classifier)

    elif model_type == 'inception':
        model = models.inception_v3(pretrained=True)
        # models.fc: 2048 -> 1000
        classifier = nn.Linear(in_features=model.fc.out_features, out_features=num_classes, bias=True)  # 1000
        model = nn.Sequential(model, classifier)

    elif model_type == 'googlenet':
        model = models.googlenet(pretrained=True)
        # models.fc: 1024 -> 1000
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=True)     # 1024

    elif model_type == 'wideresnet50':
        model = models.wide_resnet50_2(pretrained=True)
        # models.fc: 2048 -> 1000
        fc = nn.Linear(in_features=model.fc.out_features, out_features=256, bias=True)   # 1000
        classifier = nn.Linear(in_features=256, out_features=num_classes, bias=True)   # 1000
        model = nn.Sequential(model, fc, nn.Dropout(0.5), classifier)

    else:
        print("ERROR: Invalid name of the type of model")

    return model


def get_transform(augmentation):
    #if mode == 'train':
    if augmentation == 4 :
        transformer = {
            'base': transforms.Compose([
                #transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'flip' : transforms.Compose([
                #transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(0.9),
                transforms.RandomVerticalFlip(0.9),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'affine': transforms.Compose([
                #transforms.Resize(224),
                transforms.RandomAffine(degrees=(-70,70), scale=(1.2, 1.2)),
                #transforms.RandomHorizontalFlip(0.3),
                #transforms.RandomVerticalFlip(0.3),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'blur': transforms.Compose([
                #transforms.Resize(224),
                transforms.GaussianBlur(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    else:
        transformer = {
            'base': transforms.Compose([
                #transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'flip' : transforms.Compose([
                #transforms.Resize(224),
                #transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'affine': transforms.Compose([
                #transforms.Resize(224),
                transforms.RandomAffine(degrees=(-70,70), scale=(1.2, 1.2)),
                #transforms.RandomHorizontalFlip(0.3),
                #transforms.RandomVerticalFlip(0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'rotation': transforms.Compose([
                #transforms.Resize(224),
                transforms.RandomRotation(degrees=(-50,50)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'blur': transforms.Compose([
                #transforms.Resize(224),
                transforms.GaussianBlur(3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        #print("### augmentation.py: Invalid mode is given.")

    return transformer


def cutout(img, min_box=25, max_box=100, ratio=4):
    imgs_cutout=img.copy()
    xspace = img.shape[0]//ratio
    yspace = img.shape[1]//ratio
    if random.random()<0.5:
        random_st_x = random.randint(0,xspace)
        random_st_y = random.randint(0,yspace)
        random_width=random.randint(min_box,xspace)
        random_hight=random.randint(min_box,xspace)
    else:
        random_st_x = random.randint(xspace*2,img.shape[0]-100)
        random_st_y = random.randint(yspace*2,img.shape[1]-100)
        random_width=random.randint(min_box,xspace)
        random_hight=random.randint(min_box,yspace)

    imgs_cutout[random_st_x:random_st_x+random_width,random_st_y:random_st_y+random_hight]=0
    return imgs_cutout


def cutout2(img, min_box=25, max_box=100, ratio=4):
    imgs_cutout=img.copy()
    xspace = img.shape[0]//ratio
    yspace = img.shape[1]//ratio

    width=random.randint(min_box,max_box)
    height=random.randint(min_box,max_box)

    if random.random()<0.5:
        start_x = random.randint(0, xspace)
        start_y = random.randint(0, yspace)
        imgs_cutout[start_x : start_x+width, start_y:start_y+height] = 0.
    else:
        end_x = random.randint(img.shape[0]-xspace, img.shape[0])
        end_y = random.randint(img.shape[1]-yspace, img.shape[1])
        imgs_cutout[end_x-width : end_x, end_y-height :end_y]=0

    return imgs_cutout

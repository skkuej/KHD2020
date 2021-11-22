import time
import numpy as np
from copy import deepcopy
import pandas as pd
import csv
import torch
from sklearn.metrics import roc_auc_score as AUC

from metrics import get_metrics, confusion_3, confusion_2, confusion_loc

basepath = '/USER/INFERENCE/CANCER/'
#PREFIX = 'best'
#PREFIX = 'aug'
#PREFIX = 'nfold'
PREFIX = 'model'


def train(model, n_epochs, trainloader, valloader, criterion, optimizer, scheduler, device, trial_no, base_epoch, prefix, skip_batch):
    if not prefix:
        prefix = PREFIX

    best_model = deepcopy(model)
    best_auc = -np.inf

    if base_epoch: val_perf = np.load(f'{basepath}val_perf/{prefix}_{trial_no}.npy')[:base_epoch+1].tolist()
    else: val_perf = []

    print_once = 0
    for epoch in range(n_epochs):
        epoch = epoch + base_epoch

        model.train()
        start_time = time.time()
        get_lr = optimizer.param_groups[0]['lr']
        print(f'\n##### n_epoch:{epoch}, lr:{get_lr} #####')

        train_loss = 0.0
        total_labels = []
        total_preds = np.array([])
        train_confusion = np.zeros((2, 2))

        if skip_batch:
            skip = np.random.choice(len(trainloader), round(3566*0.1/trainloader.batch_size), replace=False)
        else:
            skip = np.array([])
        for i,data in enumerate(trainloader, 0):
            if not np.sum(skip==i):
                inputs, labels = data['image'], data['label']
                del data
                if not print_once:
                    print('** image size: ', inputs.shape)
                    print_once = 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

                proba = torch.nn.Softmax(dim=1)(outputs)

                train_loss += loss.item()
                total_labels.extend(labels.cpu())
                total_preds = np.hstack((total_preds, np.array(proba.detach().cpu())[:,1]))
                train_confusion += confusion_2(np.array(proba.detach().cpu()), np.array(labels.cpu()))

                del inputs, labels
        duration_min = (time.time() - start_time) / 60

        train_metrics = get_metrics(np.array(total_labels), total_preds)
        del total_labels, total_preds
        print('[TRAIN] loss: %f, auc: %f, acc: %f, f1: %f'
                        %(train_loss/len(trainloader), train_metrics['auc'], train_metrics['acc'], train_metrics['f1']))
        print(train_confusion)
        print(f'process_time : {duration_min}')
        #scheduler.step()

        test_loss, test_metrics, test_confusion = test(model, valloader, criterion, device)
        #scheduler.step(test_metrics['auc'])
        print('[VALIDATION] loss: %f, auc: %f, acc: %f, f1: %f'
                        %(test_loss, test_metrics['auc'], test_metrics['acc'], test_metrics['f1']))
        print(test_confusion)
        val_perf.append([epoch, test_loss, test_metrics['auc'], test_metrics['acc'], test_metrics['f1'], test_confusion[1,0], test_confusion[0,1], test_confusion[1,0]+test_confusion[0,1]])

        torch.save(model.state_dict(), f'{basepath}weights/{prefix}{trial_no}_epoch{epoch}.pth')
        print(f'Saved ----- {prefix}{trial_no}_epoch{epoch}.pth')
        np.save(f'{basepath}val_perf/{prefix}_{trial_no}.npy', np.array(val_perf))

        #test_auc = sudo_infer(model, testloader, device)

        if test_metrics['auc'] > best_auc:
            best_model = deepcopy(model)
    torch.save(best_model.state_dict(), f'{basepath}weights/{prefix}{trial_no}_best_epoch{epoch}.pth')

    return best_model#, np.array(val_perf)


def test(model, data_loader, criterion, device):
    model.eval()
    total_loss=0.0
    total_labels = []
    total_preds = np.array([])
    test_confusion = np.zeros((2, 2))

    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, labels = data['image'], data['label']
            del data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            proba = torch.nn.Softmax(dim=1)(outputs)

            total_labels.extend(labels.cpu())
            total_preds = np.hstack((total_preds, np.array(proba.cpu())[:,1]))

            #test_confusion += confusion_3(np.array(proba.cpu()), np.array(targets))
            test_confusion += confusion_2(np.array(proba.cpu()), np.array(labels.cpu()))
            #test_confusion += confusion_loc(np.array(proba.cpu()), np.array(labels.cpu()), np.array(location))

    del inputs, labels
    test_metrics = get_metrics(np.array(total_labels), total_preds)
    del total_labels, total_preds
    return total_loss/len(data_loader), test_metrics, test_confusion

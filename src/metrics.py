import os
import numpy as np
from sklearn.metrics import roc_auc_score

from utils import label_to_num


def _confusion_matrix(pred, label, positive_class=1):
    '''
    (pred, label)
    TN (not p_class, not p_class) / FN (not p_class, p_class) / FP (p_class, not p_class) / TP (p_class, p_class)
    ex)
    TN (0,0) / FN (0,1)/ FP (1,0) / TP (1,1)
    '''
    TN, FN, FP, TP = 0, 0, 0, 0

    for y_hat, y in zip(pred, label):
        if y != positive_class:
            if y_hat != positive_class:
                    TN = TN + 1
            else:
                    FN = FN + 1
        elif y == positive_class:
            if y_hat != positive_class:
                FP = FP + 1
            else:
                TP = TP + 1
    return TN, FN, FP, TP


def confusion_2(pred, label):
    n_classes = 2
    confusion = np.zeros((n_classes, n_classes))
    pred = np.argmax(pred, 1)
    #pred = np.round(pred)
    for i in range(n_classes):
        for j in range(n_classes):
            confusion[i,j] = np.sum(pred[label==i]==j)
    return confusion


def confusion_loc(pred, label, location):
    n_classes = 2
    confusion = np.zeros((n_classes, n_classes))
    total_conf = [confusion] * 4
    #total_conf = np.zeros((4, n_classes, n_classes))

    pred = np.argmax(pred, 1)
    #pred = np.round(pred)

    for loc in np.unique(location):
        confusion = np.zeros((n_classes, n_classes))
        label_i = label[location==loc]
        pred_j = pred[location==loc]
        for i in range(n_classes):
            for j in range(n_classes):
                confusion[i,j] = np.sum(pred_j[label_i==i]==j)
        total_conf[loc] = confusion

    return np.array(total_conf)


def confusion_3(pred, target):
    classes = [-1, 0, 1]
    n_classes = len(classes)
    confusion = np.zeros((n_classes, n_classes))
    pred = np.argmax(pred, 1)
    label = label_to_num(target)

    for i in classes:
        for j in classes:
            confusion[i+1,j+1] = np.sum(pred[label==i]==j)
    return confusion


def get_metrics(label, pred, num_class=2, eps=1e-5):
    '''
    label : 0,1
    pred : softmax
    '''
    #pred = np.argmax(pred, 1)
    p_class = 1

    metrics = dict()
    #num_P, num_N = np.sum(label == p_class), np.sum(label != p_class)

    metrics['auc'] = roc_auc_score(label, pred)
    pred = np.round(pred)
    TN, FN, FP, TP = _confusion_matrix(pred, label)
    metrics['acc'] = (TP + TN) / (TN + FN + FP + TP)
    metrics['prec'] = TP / (TP + FP + eps) ## ppv
    metrics['recall'] = TP / (TP + FN + eps) ## sensitivity
    metrics['f1'] = 2*(metrics['prec'] * metrics['recall'])/(metrics['prec'] + metrics['recall'] + eps)

    return metrics

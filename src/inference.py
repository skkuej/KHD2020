import sys, re
import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score

basepath = '/USER/INFERENCE/CANCER/'
sys.path.append(basepath + 'src/')
from dataloader_cancer import ImageDataset
from utils import seed_everything, my_model

seed = 20
seed_everything(seed)

def infer(model, data_loader, device):
    codes = []
    preds = np.array([])

    print_once = 0
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(data_loader, 0):
            inputs, code = data['image'], data['code']
            if not print_once:
                print('** image size: ', inputs.shape)
                print_once = 1
            inputs = inputs.to(device)

            outputs = model.forward(inputs)
            proba = torch.nn.Softmax(dim=1)(outputs)

            if i%100 == 99:
                print(i+1)

            codes.extend(code)
            preds = np.hstack((preds, np.array(proba.cpu())[:,1]))

    return codes, preds


def get_transform(option):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if option == 0:
        transformer = transforms.Compose([transforms.ToTensor(), normalize])
    elif option == 1:
        transformer = transforms.Compose([transforms.RandomHorizontalFlip(1), transforms.ToTensor(), normalize])
    elif option == 2:
        transformer = transforms.Compose([transforms.RandomVerticalFlip(1), transforms.ToTensor(), normalize])
    return transformer


def main(filename, savename, model_type):
    print('** Input filenames: ', filename)
    num_model = len(filename)

    if model_type!=None and num_model != len(model_type):
        print('## ERROR: number of filenames and model types are not matched!')
    if model_type == None:
        print('## No model type is defined.')
        print('## --> Set as default (efficientnet-b0)')
        model_type = [0]*num_model

    model_names = ['efficientnet-b0', 'efficientnet-b5']

    print('** Model types: ', model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformer = transforms.Compose([transforms.ToTensor(), normalize])

    averaged = []
    for i in range(len(filename)):
        fn = filename[i]
        mtype = model_names[model_type[i]]
        print(f'\n{i+1}/{num_model}... infer ... {fn}.pth ... {mtype}')
        if mtype == 'efficientnet-b0': small_size = 0
        else: small_size = 1

        test_dataset = ImageDataset(mode='test', transform=transformer, smaller=small_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = my_model(3, 2, mtype)
        model.load_state_dict(torch.load(f'{basepath}weights/{fn}.pth'))
        model.to(device)

        code, pred = infer(model, test_loader, device)

        averaged.append(pred)

        pd.DataFrame({'filename': code, 'pred': pred}).to_csv(f'{basepath}results/{fn}.csv', index=False)

    if len(filename) > 1:
        averaged = np.mean(np.array(averaged), axis=0)

        print(averaged)
        pd.DataFrame({'filename': code, 'pred': averaged}).to_csv(f'{basepath}results/{savename}.csv', index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "Step2_ Write result file")
    parser.add_argument(dest='filenames', type=str, nargs='+', help="Filename for model.pth and result.csv")
    parser.add_argument('-m', dest='model_types', type=int, nargs='+', required=False, help="The type of the model")
    parser.add_argument('-s', dest='savename', type=str, required=False, help="Filename to save in case of ensemble")
    #parser.add_argument('-s', dest='small_size', type=int, nargs='+', required=False, help="Downscaling input")
    args = parser.parse_args()

    main(args.filenames, args.savename, args.model_types)#, args.small_size)


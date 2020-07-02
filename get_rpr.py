import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_layer', type=str, default='22')
    parser.add_argument('--batchSize', type=int, default=2500)
    params = parser.parse_args()

    model_path = 'models/vgg16.pth'
    data_dir = 'data/cifar10/'
    resp_dir = 'resps/vgg16/'

    if not os.path.exists(resp_dir):
        os.makedirs(resp_dir)

    device = torch.device('cuda')

    # model
    model1 = model.VGG()
    model1.load_state_dict(torch.load(model_path))
    model1.to(device)
    model1.eval()
    model2 = model.VGG2(model1, params.target_layer)

    for state in ['test', 'train']:
        # data
        transform = tfs.Compose([tfs.ToTensor()])
        data = dst.CIFAR10(data_dir, download=True,
                           train=(state=='train'),
                           transform=transform)
        dataloader = DataLoader(data, batch_size=params.batchSize, shuffle=False)

        with torch.no_grad():
            for counter, (data, target) in enumerate(dataloader):
                y = model2.forward(data.to(device))
                fname = resp_dir + state + '_' + str(params.target_layer) + \
                        '_' + str(counter)
                np.save(fname, y.detach().cpu().numpy())

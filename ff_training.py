import os
import time
import sys

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model

def train(dataloader, net):
    net.train()
    total = 0
    correct = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = net(x)
        lossv = loss_f(output, y)
        lossv.backward()
        optimizer.step()
        correct += y.eq(torch.max(output.data, 1)[1]).sum().item()
        total += y.numel()
    return correct / total

def test(dataloader, net):
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = net(x)
            correct += y.eq(torch.max(output.data, 1)[1]).sum().item()
            total += y.numel()
    return correct / total

if __name__ == "__main__":
    # parameters
    batchSize = 128
    lr = 1e-4
    model_path = 'models/vgg16.pth'
    data_dir = 'data/cifar10/'
    epochs = [80, 20]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists('models'):
        os.makedirs('models')

    device = torch.device('cuda')

    # model and loss
    net = model.VGG()
    loss_f = nn.CrossEntropyLoss()
    net.to(device)
    loss_f.to(device)

    # data
    transform_train = tfs.Compose([tfs.RandomCrop(32, padding=4),
                                   tfs.RandomHorizontalFlip(),
                                   tfs.ToTensor()])
    data = dst.CIFAR10(data_dir, download=True, train=True,
                       transform=transform_train)
    data_test = dst.CIFAR10(data_dir, download=True, train=False,
                            transform=tfs.Compose([tfs.ToTensor()]))
    dataloader = DataLoader(data, batch_size=batchSize, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=batchSize, shuffle=False)

    count = 0
    for epoch in epochs:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        for _ in range(epoch):
            beg = time.time()
            count += 1
            train_acc = train(dataloader, net)
            test_acc = test(dataloader_test, net)
            run_time = time.time() - beg
            print('Epoch {}, Time {:.2f}, Train: {:.5f}, Test: {:.5f}'.\
                  format(count, run_time, train_acc, test_acc))
            sys.stdout.flush()

        lr /= 10

    torch.save(net.state_dict(), model_path)

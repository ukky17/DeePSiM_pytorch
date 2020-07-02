import os
import time
import argparse
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.datasets as dst
import torchvision.transforms as tfs
from torch.utils.data import DataLoader

import model

# define the data_loader
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        if self.transform:
            out_data = self.transform(out_data)
        return out_data, out_label

def train(dataloader, epoch, params):
    net_c.eval()
    model1.eval()

    std = (1 / 256.0) * (1 - epoch / 256.0)
    cum_loss = [0] * 5

    for batch_idx, (activations, real_image) in enumerate(dataloader):
        activations = activations.type('torch.FloatTensor').to(device)
        real_image = real_image.type('torch.FloatTensor').to(device)

        # update Generator
        optimizer_g.zero_grad()
        net_g.train()
        net_d.eval()

        real_l = torch.argmax(model1(real_image), 1)
        fake_image = net_g(activations)
        fake_logit = net_d(fake_image, std)
        fake_feature = net_c(fake_image)
        real_feature = net_c(real_image)

        loss_feat = loss_f_feat(real_feature, fake_feature)
        loss_img = loss_f_img(real_image, fake_image)

        if params.gan_type == 'lsgan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]

            loss_f_adv = nn.MSELoss()
            loss_f_adv.to(device)
            loss_adv = loss_f_adv(fake_logit, ones)
        elif params.gan_type == 'gan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]

            loss_f_adv = nn.BCEWithLogitsLoss()
            loss_f_adv.to(device)
            loss_adv = loss_f_adv(fake_logit, ones)
        elif params.gan_type == 'wgan':
            loss_adv = -torch.mean(fake_logit)

        loss_g = params.lambda_feat * loss_feat + \
                 params.lambda_adv * loss_adv + \
                 params.lambda_img * loss_img

        loss_g.backward()
        optimizer_g.step()

        # update Discriminator
        optimizer_d.zero_grad()
        net_g.eval()
        net_d.train()

        fake_logit = net_d(fake_image.detach(), std)
        real_logit = net_d(real_image, std)

        if params.gan_type == 'lsgan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]
            zeros = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.zeros(real_image.shape[0])]

            loss_f_adv = nn.MSELoss()
            loss_f_adv.to(device)
            loss_d = loss_f_adv(real_logit, ones) + loss_f_adv(fake_logit, zeros)
        elif params.gan_type == 'gan':
            ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.ones(real_image.shape[0])]
            zeros = torch.eye(2, dtype=torch.float32, requires_grad=False,
                             device=device)[np.zeros(real_image.shape[0])]

            loss_f_adv = nn.BCEWithLogitsLoss()
            loss_f_adv.to(device)
            loss_d = loss_f_adv(real_logit, ones) + loss_f_adv(fake_logit, zeros)
        elif params.gan_type == 'wgan':
            loss_d = -torch.mean(real_logit) + torch.mean(fake_logit)

        if not (params.apply_th and loss_d.item() < 0.1 * loss_adv.item()):
            loss_d.backward()
            optimizer_d.step()

        if params.gan_type == 'wgan':
            # clip parameters
            for param in net_d.parameters():
                param.data.clamp_(-params.clip_value, params.clip_value)

        loss_list = [loss_img, loss_adv, loss_feat, loss_g, loss_d]
        for i, _loss in enumerate(loss_list):
            cum_loss[i] += _loss.item()

    for i in range(len(cum_loss)):
        cum_loss[i] /= (batch_idx + 1)
    return cum_loss

def test(dataloader, epoch, params):
    net_g.eval()
    net_d.eval()
    net_c.eval()
    model1.eval()

    std = 0.0
    cum_loss = [0] * 5

    with torch.no_grad():
        for batch_idx, (activations, real_image) in enumerate(dataloader):
            activations = activations.type('torch.FloatTensor').to(device)
            real_image = real_image.type('torch.FloatTensor').to(device)

            real_l = torch.argmax(model1(real_image), 1)
            fake_image = net_g(activations)
            fake_logit = net_d(fake_image, std)
            real_logit = net_d(real_image, std)
            fake_feature = net_c(fake_image)
            real_feature = net_c(real_image)

            # generator loss
            loss_feat = loss_f_feat(real_feature, fake_feature)
            loss_img = loss_f_img(real_image, fake_image)

            if params.gan_type == 'lsgan':
                ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.ones(real_image.shape[0])]

                loss_f_adv = nn.MSELoss()
                loss_f_adv.to(device)
                loss_adv = loss_f_adv(fake_logit, ones)
            elif params.gan_type == 'gan':
                ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.ones(real_image.shape[0])]

                loss_f_adv = nn.BCEWithLogitsLoss()
                loss_f_adv.to(device)
                loss_adv = loss_f_adv(fake_logit, ones)
            elif params.gan_type == 'wgan':
                loss_adv = -torch.mean(fake_logit)

            loss_g = params.lambda_feat * loss_feat + \
                     params.lambda_adv * loss_adv + \
                     params.lambda_img * loss_img

            # discriminator loss
            if params.gan_type == 'lsgan':
                ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.ones(real_image.shape[0])]
                zeros = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.zeros(real_image.shape[0])]

                loss_f_adv = nn.MSELoss()
                loss_f_adv.to(device)
                loss_d = loss_f_adv(real_logit, ones) + loss_f_adv(fake_logit, zeros)
            elif params.gan_type == 'gan':
                ones = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.ones(real_image.shape[0])]
                zeros = torch.eye(2, dtype=torch.float32, requires_grad=False,
                                 device=device)[np.zeros(real_image.shape[0])]

                loss_f_adv = nn.BCEWithLogitsLoss()
                loss_f_adv.to(device)
                loss_d = loss_f_adv(real_logit, ones) + loss_f_adv(fake_logit, zeros)
            elif params.gan_type == 'wgan':
                loss_d = -torch.mean(real_logit) + torch.mean(fake_logit)

            loss_list = [loss_img, loss_adv, loss_feat, loss_g, loss_d]
            for i, _loss in enumerate(loss_list):
                cum_loss[i] += _loss.item()

    # plot
    real_image = real_image.detach().cpu().numpy()
    fake_image = fake_image.detach().cpu().numpy()
    plt.figure(figsize=(12, 12))
    for i in range(min(50, len(real_image))):
        img1 = np.transpose(real_image[i] * 255, (1, 2, 0)).astype('uint8')
        ax = plt.subplot(10, 10, 2 * i + 1)
        ax.imshow(img1)
        ax.set_title('Real')
        plt.axis('off')

        img2 = np.transpose(fake_image[i] * 255, (1, 2, 0)).astype('uint8')
        ax = plt.subplot(10, 10, 2 * i + 2)
        ax.imshow(img2)
        ax.set_title('Generated')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_dir + 'img_epoch' + str(epoch) + '.png')
    plt.close()

    for i in range(len(cum_loss)):
        cum_loss[i] /= (batch_idx + 1)
    return cum_loss


if __name__ == "__main__":
    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_layer', type=str, default='22')
    parser.add_argument('--epochs', default=256, type=int)
    parser.add_argument('--trainBatch', default=16, type=int)
    parser.add_argument('--testBatch', default=16, type=int)
    parser.add_argument('--lambda_feat', default=0.01, type=float)
    parser.add_argument('--lambda_adv', default=0.001, type=float)
    parser.add_argument('--lambda_img', default=1, type=float)
    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--lr_decay', default=0.96, type=float)
    parser.add_argument('--beta1', default=0.5, type=float)
    parser.add_argument('--beta2', default=0.99, type=float)
    parser.add_argument('--gan_type', default='lsgan', type=str)
    parser.add_argument('--optimizer_type', default='Adam', type=str)
    parser.add_argument('--apply_th', action='store_true')
    parser.add_argument('--clip_value', default=0.05, type=float)
    params = parser.parse_args()

    data_dir = 'data/cifar10/'
    resp_dir = 'resps/vgg16/'
    save_dir = 'generated/vgg16/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device = torch.device('cuda')

    # load the images and responses
    images = dict()
    resps = dict()
    for state in ['test', 'train']:
        # response
        files = glob(resp_dir + state + '_' + \
                     str(params.target_layer) + '_*.npy')

        for i, file in enumerate(files):
            if i == 0:
                _resps = np.load(file)
            else:
                _resps = np.vstack((_resps, np.load(file)))
        print(state + ' resp shape: ' + str(_resps.shape))
        resps[state] = _resps

        # image
        data = dst.CIFAR10(data_dir, download=False,
                           train=(state=='train'),
                           transform=tfs.ToTensor())

        dataloader = DataLoader(data, batch_size=len(data), shuffle=False)
        _images = next(iter(dataloader))[0].numpy()
        print(state + ' image shape: ' + str(_images.shape))
        images[state] = _images

    # data
    train_dataset = CustomDataset(resps['train'], images['train'])
    test_dataset = CustomDataset(resps['test'], images['test'])
    dataloader = DataLoader(train_dataset, batch_size=params.trainBatch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=params.testBatch, shuffle=False)

    # model
    net_g = model.Generator()
    net_g.to(device)
    net_d = model.Discriminator()
    net_d.to(device)
    model1 = model.VGG()
    model1.load_state_dict(torch.load('models/vgg16.pth'))
    model1.to(device)
    net_c = model.Comparator(model1, 22)
    net_c.to(device)
    for param in model1.parameters():
        param.requires_grad = False
    for param in net_c.parameters():
        param.requires_grad = False

    # optimizer
    if params.optimizer_type == 'Adam':
        optimizer_g = optim.Adam(net_g.parameters(), lr=params.lr,
                                 betas=(params.beta1, params.beta2))
        optimizer_d = optim.Adam(net_d.parameters(), lr=params.lr,
                                 betas=(params.beta1, params.beta2))
    elif params.optimizer_type == 'RMSprop':
        optimizer_g = optim.RMSprop(net_g.parameters(), lr=params.lr,
                                    alpha=params.beta1)
        optimizer_d = optim.RMSprop(net_d.parameters(), lr=params.lr,
                                    alpha=params.beta1)

    loss_f_img = nn.MSELoss()
    loss_f_img.to(device)
    loss_f_feat = nn.MSELoss()
    loss_f_feat.to(device)

    loss_names = ['img', 'adv', 'feat', 'g', 'd']
    losses = np.zeros((2, params.epochs, len(loss_names)))
    for epoch in range(params.epochs):
        start = time.time()
        l1 = train(dataloader, epoch, params)
        l2 = test(test_loader, epoch, params)
        run_time = time.time() - start

        losses[0, epoch, :], losses[1, epoch, :] = l1, l2
        print('Epoch {}, Time {:.2f}'.format(epoch, run_time))
        print('  Train: {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(\
                l1[0], l1[1], l1[2], l1[3], l1[4]))
        print('  Test:  {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(\
                l2[0], l2[1], l2[2], l2[3], l2[4]))

        if epoch % 128 == 0:
            params.lr *= params.lr_decay

    # save
    np.save(save_dir + 'loss_layer' + str(params.target_layer), losses)
    torch.save(net_g.state_dict(), save_dir + 'net_g.pth')

import torch
import torch.nn as nn
from torch.autograd import Variable


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
       512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.classifier = nn.Linear(512, 10)
        self.features = self._make_layers(cfg)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# input --> hidden layer
class VGG2(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG2, self).__init__()
        features = list(model1.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        if self.target_layer == -1:
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.target_layer:
                return x

# hidden layer --> output layer
class VGG3(torch.nn.Module):
    def __init__(self, model1, target_layer):
        super(VGG3, self).__init__()
        features = list(model1.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)
        self.classifier = model1.classifier

    def forward(self, x):
        for ii, model in enumerate(self.features):
            if ii > self.target_layer:
                x = model(x)

        out = x.view(x.size(0), -1)
        out = self.classifier(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            )

    def forward(self, x):
        return 0.5 * (1 + self.upsample(x))

class Comparator(nn.Module):
    def __init__(self, model1, target_layer):
        super(Comparator, self).__init__()
        features = list(model1.features)
        self.features = nn.ModuleList(features).eval()
        self.target_layer = int(target_layer)

    def forward(self, x):
        if self.target_layer == -1:
            return x

        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == self.target_layer:
                return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, std):
        noise = torch.randn(x.size(), requires_grad=False, device=torch.device('cuda')) * std
        x2 = x + noise

        x2 = self.features1(x2)
        x2 = self.avgpool(x2)
        h = torch.flatten(x2, 1)
        h = self.classifier(h)
        return h

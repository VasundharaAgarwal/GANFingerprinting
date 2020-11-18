import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

nnodes = 128*7*7
nz = 100

class Generator1(nn.Module):
  def __init__(self, ngpu):
      super(Generator1, self).__init__()
      self.ngpu = ngpu
      self.linear = nn.Linear(nz, nnodes, bias=True)
      self.leakyReLU1 = nn.LeakyReLU(0.2)
      self.convTr1 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False)
      self.leakyReLU2 = nn.LeakyReLU(0.2)
      self.convTr2 = nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False)
      self.leakyReLU3 = nn.LeakyReLU(0.2)
      self.conv1 = nn.Conv2d(128, 1, 7, 1, 3, bias=False)

  def forward(self, x):
      x = self.linear(x.view(-1,1,1,100))
      x = self.leakyReLU1(x)
      x = x.view(-1,128, 7, 7)
      x = self.leakyReLU2(self.convTr1(x))
      x = self.leakyReLU3(self.convTr2(x))
      x = torch.sigmoid(self.conv1(x))
      return x
      
class Generator2(nn.Module):
  def __init__(self, ngpu):
    super(Generator2, self).__init__()
    self.ngpu = ngpu
    self.linear = nn.Linear(nz, nnodes, bias=True)
    self.leakyReLU1 = nn.LeakyReLU(0.2)
    self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
    self.conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False )
    self.leakyReLU2 = nn.LeakyReLU(0.2)
    self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False )
    self.leakyReLU3 = nn.LeakyReLU(0.2)
    self.conv3 = nn.Conv2d(128, 1, 7, 1, 3, bias=False)

  def forward(self, x):
    x = self.linear(x.view(-1,1,1,100))
    x = self.leakyReLU1(x)
    x = x.view(-1, 128, 7, 7)
    x = self.leakyReLU2(self.conv1(self.upsample(x)))
    x = self.leakyReLU3(self.conv2(self.upsample(x)))
    x = torch.sigmoid(self.conv3(x))
    return x

class Generator3(nn.Module):
  def __init__(self, ngpu):
    super(Generator3, self).__init__()
    self.ngpu = ngpu
    self.linear = nn.Linear(nz, nnodes, bias=True)
    self.leakyReLU1 = nn.LeakyReLU(0.2)
    self.upsample = nn.Upsample(scale_factor=2)
    self.conv1 = nn.Conv2d(128, 64, 3, 1, 1, bias=False )
    self.leakyReLU2 = nn.LeakyReLU(0.2)
    self.conv2 = nn.Conv2d(64, 32, 3, 1, 1, bias=False )
    self.leakyReLU3 = nn.LeakyReLU(0.2)
    self.conv3 = nn.Conv2d(32, 1, 7, 1, 3, bias=False)

  def forward(self, x):
    x = self.linear(x.view(-1,1,1,100))
    x = self.leakyReLU1(x)
    x = x.view(-1, 128, 7, 7)
    x = self.leakyReLU2(self.conv1(self.upsample(x)))
    x = self.leakyReLU3(self.conv2(self.upsample(x)))
    x = torch.sigmoid(self.conv3(x))
    return x

class Discriminator1(nn.Module):
  def __init__(self, ngpu):
      super(Discriminator1, self).__init__()
      self.ngpu = ngpu
      self.conv1 = nn.Conv2d(1, 64, 3, 2, 1, bias=False)
      self.leakyReLU1 = nn.LeakyReLU(0.2)
      self.dropout1 = nn.Dropout(0.4)
      self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
      self.leakyReLU2 = nn.LeakyReLU(0.2)
      self.dropout2 = nn.Dropout(0.4)
      self.flatten = nn.Flatten()
      self.linear = nn.Linear(3136, 1, bias=True)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
      x = self.conv1(x)
      x = self.dropout1(self.leakyReLU1(x))
      x = self.conv2(x)
      x = self.dropout2(self.leakyReLU2(x))
      x = self.sigmoid(self.linear(self.flatten(x)))
      return x

class Discriminator2(nn.Module):
  def __init__(self, ngpu):
      super(Discriminator2, self).__init__()
      self.ngpu = ngpu
      self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
      self.maxpool2d = nn.MaxPool2d(3, 2, 1)
      self.leakyReLU1 = nn.LeakyReLU(0.2)
      self.dropout1 = nn.Dropout(0.4)
      self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
      self.leakyReLU2 = nn.LeakyReLU(0.2)
      self.dropout2 = nn.Dropout(0.4)
      self.flatten = nn.Flatten()
      self.linear = nn.Linear(3136, 1, bias=True)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
      x = self.maxpool2d(self.conv1(x))
      x = self.dropout1(self.leakyReLU1(x))
      x = self.maxpool2d(self.conv2(x))
      x = self.dropout2(self.leakyReLU2(x))
      x = self.sigmoid(self.linear(self.flatten(x)))
      return x

class Discriminator3(nn.Module):
  def __init__(self, ngpu):
      super(Discriminator3, self).__init__()
      self.ngpu = ngpu
      self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
      self.avgpool2d = nn.AvgPool2d(3, 2, 1)
      self.leakyReLU1 = nn.LeakyReLU(0.2)
      self.dropout1 = nn.Dropout(0.4)
      self.conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
      self.leakyReLU2 = nn.LeakyReLU(0.2)
      self.dropout2 = nn.Dropout(0.4)
      self.flatten = nn.Flatten()
      self.linear = nn.Linear(3136, 1, bias=True)
      self.sigmoid = nn.Sigmoid()

  def forward(self, x):
      x = self.avgpool2d(self.conv1(x))
      x = self.dropout1(self.leakyReLU1(x))
      x = self.avgpool2d(self.conv2(x))
      x = self.dropout2(self.leakyReLU2(x))
      x = self.sigmoid(self.linear(self.flatten(x)))
      return x

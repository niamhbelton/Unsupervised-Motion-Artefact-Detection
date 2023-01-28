import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F




class CIFAR_VGG3(nn.Module):
  def __init__(self,vector_size):
      super(CIFAR_VGG3, self).__init__()

      self.act = nn.LeakyReLU()
      self.block1=models.vgg16(pretrained = True).features[0]
      self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

      self.block2=models.vgg16().features[2]
      self.block3=models.vgg16().features[4:6]
      self.block4=models.vgg16().features[7]
      self.block5=models.vgg16().features[9:11]
      self.block6=models.vgg16().features[12]
      self.block7=models.vgg16().features[14]
      self.block8=models.vgg16().features[16]
      self.classifier = nn.Linear(4096, vector_size,bias=False)

  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x= self.block1(x)
      x=self.bn1(x)
      x=self.act(x)
      x= self.block2(x)
      x=self.act(x)
      x= self.block3(x)
      x=self.act(x)
      x= self.block4(x)
      x=self.act(x)
      x= self.block5(x)
      x=self.act(x)
      x= self.block6(x)
      x=self.act(x)
      x= self.block7(x)
      x=self.act(x)
      x= self.block8(x)
      x=self.act(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      x=nn.Sigmoid()(x)
      return x #output



class CIFAR_VGG3_pre(nn.Module):
  def __init__(self,vector_size):
      super(CIFAR_VGG3_pre, self).__init__()

      self.act = nn.LeakyReLU()
      self.block1=models.vgg16(pretrained = True).features[0]
      self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

      self.block2=models.vgg16(pretrained = True).features[2]
      self.block3=models.vgg16(pretrained = True).features[4:6]
      self.block4=models.vgg16(pretrained = True).features[7]
      self.block5=models.vgg16(pretrained = True).features[9:11]
      self.block6=models.vgg16(pretrained = True).features[12]
      self.block7=models.vgg16(pretrained = True).features[14]
      self.block8=models.vgg16(pretrained = True).features[16]
      self.classifier = nn.Linear(4096, vector_size,bias=False)


  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x= self.block1(x)
      x=self.bn1(x)
      x=self.act(x)
      x= self.block2(x)
      x=self.act(x)
      x= self.block3(x)
      x=self.act(x)
      x= self.block4(x)
      x=self.act(x)
      x= self.block5(x)
      x=self.act(x)
      x= self.block6(x)
      x=self.act(x)
      x= self.block7(x)
      x=self.act(x)
      x= self.block8(x)
      x=self.act(x)


      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      x=nn.Sigmoid()(x)

      return x #output


class MNIST_VGG3(nn.Module):
  def __init__(self,vector_size):
      super(MNIST_VGG3, self).__init__()

      self.act = nn.LeakyReLU()
      self.block1=models.vgg16().features[0]
      self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

      self.block2=models.vgg16().features[2]
      self.block3=models.vgg16().features[4:6]
      self.block4=models.vgg16().features[7]
      self.block5=models.vgg16().features[9:11]
      self.block6=models.vgg16().features[12]
      self.block7=models.vgg16().features[14]
      self.block8=models.vgg16().features[16]
      self.classifier = nn.Linear(2304, vector_size,bias=False)

  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x= self.block1(x)
      x=self.bn1(x)
      x=self.act(x)
      x= self.block2(x)
      x=self.act(x)
      x= self.block3(x)
      x=self.act(x)
      x= self.block4(x)
      x=self.act(x)
      x= self.block5(x)
      x=self.act(x)
      x= self.block6(x)
      x=self.act(x)
      x= self.block7(x)
      x=self.act(x)
      x= self.block8(x)
      x=self.act(x)


      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      x=nn.Sigmoid()(x)

      return x #output





class MNIST_VGG3_pre(nn.Module):
  def __init__(self,vector_size):
      super(MNIST_VGG3_pre, self).__init__()

      self.act = nn.LeakyReLU()
      self.block1=models.vgg16(pretrained = True).features[0]
      self.bn1 = nn.BatchNorm2d(64, eps=1e-04, affine=False)

      self.block2=models.vgg16(pretrained = True).features[2]
      self.block3=models.vgg16(pretrained = True).features[4:6]
      self.block4=models.vgg16(pretrained = True).features[7]
      self.block5=models.vgg16(pretrained = True).features[9:11]
      self.block6=models.vgg16(pretrained = True).features[12]
      self.block7=models.vgg16(pretrained = True).features[14]
      self.block8=models.vgg16(pretrained = True).features[16]
      self.classifier = nn.Linear(2304, vector_size,bias=False)


  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x= self.block1(x)
      x=self.bn1(x)
      x=self.act(x)
      x= self.block2(x)
      x=self.act(x)
      x= self.block3(x)
      x=self.act(x)
      x= self.block4(x)
      x=self.act(x)
      x= self.block5(x)
      x=self.act(x)
      x= self.block6(x)
      x=self.act(x)
      x= self.block7(x)
      x=self.act(x)
      x= self.block8(x)
      x=self.act(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      x=nn.Sigmoid()(x)

      return x #output

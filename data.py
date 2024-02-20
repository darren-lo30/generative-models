import torchvision
import torch
from torchvision import transforms

def split_dataset(dataset, valid_ratio):
  train_cnt = int(len(dataset) * (1 - valid_ratio))
  valid_cnt = len(dataset) - train_cnt
  train_set, valid_set = torch.utils.data.random_split(dataset, [train_cnt, valid_cnt])

  return train_set, valid_set

def load_mnist(valid_ratio = 0.1, batch_size=256, add_transforms = []):
  transform = transforms.Compose([transforms.ToTensor()] + add_transforms)
  dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
  train_set, valid_set = split_dataset(dataset, valid_ratio)
  train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
  valid_iter = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=False)

  return train_iter, valid_iter

def load_cifar_10(valid_ratio = 0.1, batch_size=256):
  transform = transforms.Compose([transforms.ToTensor()] + transforms)
  dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
  train_set, valid_set = split_dataset(dataset, valid_ratio)
  train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True)
  valid_iter = torch.utils.data.DataLoader(valid_set, batch_size, shuffle=False)

  return train_iter, valid_iter
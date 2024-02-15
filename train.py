import torchvision
import torch
import vae
from torchvision import transforms
from matplotlib import pyplot as plt
from utils import get_device

def load_data():
  transform = transforms.Compose([transforms.ToTensor()])
  dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
  train_set, valid_set = torch.utils.data.random_split(dataset, [50000, 10000])
  train_iter = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
  valid_iter = torch.utils.data.DataLoader(valid_set, batch_size=256, shuffle=True)

  return train_iter, valid_iter

def visualize_out(imgs):
  count = len(imgs)
  num_col = 5
  num_row = count // num_col
  fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
  for i in range(count):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(imgs[i], cmap='gray')
  plt.tight_layout()
  plt.show()
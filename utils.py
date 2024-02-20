import torch
from torch import nn
from matplotlib import pyplot as plt
import math
from torchvision import transforms

def get_device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ELBOLoss(mu, log_var, x_hat, x):
  reconstruction = -nn.functional.mse_loss(x_hat, x, reduction='sum')
  # This is the KL divergence between Q(z|x) and N(0, I)
  divergence = 0.5 * (log_var.exp() + mu.pow(2) - 1 - log_var).sum() 

  elbo = reconstruction - divergence
  # Return negative since we want to maximize
  return -elbo


def visualize_out(imgs, cmap='gray'):
  count = len(imgs)
  num_col = 5
  num_row = math.ceil(count / num_col)
  fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row), squeeze=False)
  for i in range(count):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(imgs[i], cmap=cmap)
  plt.tight_layout()
  plt.show()

def to_pil(imgs):
  transform = transforms.ToPILImage()
  return [transform(img) for img in imgs]
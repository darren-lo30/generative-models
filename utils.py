import torch
from torch import nn

def get_device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ELBOLoss(mu, log_var, x_hat, x):
  reconstruction = -nn.functional.mse_loss(x_hat, x, reduction='sum')
  # This is the KL divergence between Q(z|x) and N(0, I)
  divergence = 0.5 * (log_var.exp() + mu.pow(2) - 1 - log_var).sum() 

  elbo = reconstruction - divergence
  # Return negative since we want to maximize
  return -elbo

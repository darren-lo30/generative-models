from torch import nn
import torch

class MLPEncoder(nn.Module):
  def __init__(self, layer_sizes, latent_dim):
    super(MLPEncoder, self).__init__()
    self.net = nn.Sequential()
    for i, layer_size in enumerate(layer_sizes):
      self.net.add_module(f"layer_{i}", nn.Sequential(nn.LazyLinear(layer_size), nn.ReLU()))

    self.mu_layer = nn.LazyLinear(latent_dim)
    self.sigma_layer = nn.LazyLinear(latent_dim)
    

  def forward(self, X):
    out = self.net(X)
    # (batch_size, 2) where first is mu, second is sigma
    mu, log_var = self.mu_layer(out), self.sigma_layer(out)

    return mu, log_var
# class Decoder(nn.Module):

class MLPDecoder(nn.Module):
  def __init__(self, layer_sizes, out_dim):
    super(MLPDecoder, self).__init__()
    
    self.net = nn.Sequential()
    for i, layer_size in enumerate(layer_sizes):
      self.net.add_module(f"layer_{i}", nn.Sequential(nn.LazyLinear(layer_size), nn.ReLU()))

    self.net.add_module("last_layer", nn.Sequential(nn.LazyLinear(out_dim), nn.Sigmoid()))
    
  def forward(self, z):
    return self.net(z)

class VAE(nn.Module):
  def __init__(self, encoder, decoder):
    super(VAE, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  # epsilon ~ N(0, I) (batch_size, 1)
  def forward(self, x, epsilon):
    # Encoder produces mu, sigma
    # z ~ N(mu, sigma^2 I) 
    mu, log_var = self.encoder(x)
    z = mu + epsilon * torch.exp(0.5 * log_var)
    
    x_hat = self.decoder(z)

    # We need log_var, mu to calculate loss
    return x_hat, mu, log_var
    

def ELBOLoss(mu, log_var, x_hat, x):
  reconstruction = -nn.functional.mse_loss(x_hat, x, reduction='sum')
  # This is the KL divergence between Q(z|x) and N(0, I)
  divergence = 0.5 * (log_var.exp() + mu.pow(2) - 1 - log_var).sum() 

  elbo = reconstruction - divergence
  # Return negative since we want to maximize
  return -elbo
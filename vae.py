from torch import nn
import torch
from utils import ELBOLoss

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
  def __init__(self, encoder, decoder, device):
    super(VAE, self).__init__()
    self.device = device
    self.encoder = encoder
    self.decoder = decoder

  # epsilon ~ N(0, I) (batch_size, 1)
  def forward(self, x):
    # Encoder produces mu, sigma
    # z ~ N(mu, sigma^2 I) 
    mu, log_var = self.encoder(x)
    epsilon = torch.randn_like(log_var).to(self.device)
    z = mu + epsilon * torch.exp(0.5 * log_var)
    
    x_hat = self.decoder(z)

    # We need log_var, mu to calculate loss
    return x_hat, mu, log_var

def train_vae(net, optimizer, train_iter, valid_iter, num_epochs = 20, device=get_device()):
  for epoch in range(num_epochs):
    avg_train_loss = 0
    for examples, _ in train_iter:
      examples = examples.to(device)
      examples = examples.reshape(examples.shape[0], -1) # reshape into (batch_size, 28 * 28) 
      optimizer.zero_grad()
      net.train()

      # epsilon ~ N(0, I)
      x_hat, mu, log_var = net(examples)

      elbo_loss = ELBOLoss(mu, log_var, x_hat, examples)

      elbo_loss.backward()
      avg_train_loss += elbo_loss.cpu().detach().numpy()
      optimizer.step()
    avg_train_loss /= len(train_iter.dataset)
    print(f'Epoch {epoch} - Train Loss: {avg_train_loss}')

    avg_valid_loss =0
    for examples, _ in valid_iter:
      net.eval()
      examples = examples.to(device)
      examples = examples.reshape(examples.shape[0], -1)
      
      x_hat, mu, log_var = net(examples)
      elbo_loss = vae.ELBOLoss(mu, log_var, x_hat, examples)

      avg_valid_loss += elbo_loss

    avg_valid_loss /= len(valid_iter.dataset)
    print(f'Epoch {epoch} - Valid Loss: {avg_valid_loss}')
from torch import nn
import torch
from utils import get_device
from utils import ELBOLoss


class ConditionalMLPEncoder(nn.Module):
  def __init__(self, layer_sizes, latent_dim):
    super(ConditionalMLPEncoder, self).__init__()
    self.net = nn.Sequential()
    for i, layer_size in enumerate(layer_sizes):
      self.net.add_module(f"layer_{i}", nn.Sequential(nn.LazyLinear(layer_size), nn.ReLU()))

    self.mu_layer = nn.LazyLinear(latent_dim)
    self.sigma_layer = nn.LazyLinear(latent_dim)
    
  # P(z | f(X, cond), g(x, cond))
  def forward(self, X, cond):
    out = self.net(X)
    out = torch.cat((out, cond), dim=1)
    mu, log_var = self.mu_layer(out), self.sigma_layer(out)

    return mu, log_var

class ConditionalMLPDecoder(nn.Module):
  def __init__(self, layer_sizes, out_dim):
    super(ConditionalMLPDecoder, self).__init__()
    
    self.net = nn.Sequential()
    for i, layer_size in enumerate(layer_sizes):
      self.net.add_module(f"layer_{i}", nn.Sequential(nn.LazyLinear(layer_size), nn.ReLU()))

    self.net.add_module("last_layer", nn.Sequential(nn.LazyLinear(out_dim), nn.Sigmoid()))
    
  def forward(self, z, cond):
    return self.net(torch.cat((z, cond), dim = 1))
  
class CVAE(nn.Module):
  def __init__(self, encoder, decoder):
    super(CVAE, self).__init__()
    self.encoder = encoder
    self.decoder = decoder

  # epsilon ~ N(0, I) (batch_size, 1)
  def forward(self, x, cond):
    # Encoder produces mu, sigma
    # z ~ N(mu, sigma^2 I) 
    mu, log_var = self.encoder(x, cond)
    # Reparameterization
    epsilon = torch.randn_like(log_var)
    z = mu + epsilon * torch.exp(0.5 * log_var)
    
    x_hat = self.decoder(z, cond)

    # We need log_var, mu to calculate loss
    return x_hat, mu, log_var
  
def train_cvae(net, optimizer, train_iter, valid_iter, num_epochs = 20, device=get_device()):
  for epoch in range(num_epochs):
    avg_train_loss = 0
    for examples, labels in train_iter:
      # (batch_size, 10)
      labels = torch.nn.functional.one_hot(labels, 10)
      labels = labels.to(device)
      
      examples = examples.to(device)
      examples = examples.reshape(examples.shape[0], -1) # reshape into (batch_size, 28 * 28) 
      optimizer.zero_grad()
      net.train()

      # epsilon ~ N(0, I)
      x_hat, mu, log_var = net(examples, labels)

      elbo_loss = ELBOLoss(mu, log_var, x_hat, examples)

      elbo_loss.backward()
      avg_train_loss += elbo_loss.cpu().detach().numpy()
      optimizer.step()
    avg_train_loss /= len(train_iter.dataset)
    print(f'Epoch {epoch} - Train Loss: {avg_train_loss}')

    avg_valid_loss =0
    for examples, labels in valid_iter:
      net.eval()
      # (batch_size, 10)
      labels = labels.to(device)      
      labels = torch.nn.functional.one_hot(labels, 10)

      examples = examples.to(device)
      examples = examples.reshape(examples.shape[0], -1)
      
      x_hat, mu, log_var = net(examples, labels)
      elbo_loss = ELBOLoss(mu, log_var, x_hat, examples)

      avg_valid_loss += elbo_loss

    avg_valid_loss /= len(valid_iter.dataset)
    print(f'Epoch {epoch} - Valid Loss: {avg_valid_loss}')
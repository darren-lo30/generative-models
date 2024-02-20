import torch
from utils import get_device, visualize_out, to_pil
from torch import nn

# Diffusion model 
class DiffusionModel(nn.Module):
  # model - used to predict the original noise epsilon_0
  # num_steps - the number of diffusion steps
  # alpha - denotes alpha_t for [0, T]
  def __init__(self, model, alpha, device=get_device()):
    super().__init__()
    self.device = device
    # alpha_0 is never used, should be 1
    self.alpha = alpha
    self.T = len(alpha) - 1
    self.model = model
    # alpha_bar[t] = alpha[0] * ... * alpha[t]
    self.alpha_bar = torch.cumprod(alpha, dim = 0)
    self.var = 1 - alpha

  def forward(self, x):
    xt, t, epsilon_0 = self.sample_xt(x)
    # Aims to predict epsilon_0
    pred = self.model(xt, t)
    return pred, epsilon_0

  # Samples t ~ [2, T]
  # Samples xt from x0, t 
  def sample_xt(self, x0):
    t = torch.randint(2, self.T, (x0.shape[0], 1)).squeeze().to(device=self.device)
    alpha_bar_t = torch.index_select(self.alpha_bar, 0, t).view((-1, 1, 1, 1))
    epsilon_0 = torch.randn_like(x0).to(device=self.device)
    xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon_0
    return xt, t, epsilon_0
  
  # x_{t-1} ~ p(x_{t-1} | x_t)
  def sample_p(self, xt, t):
    alpha_t = self.alpha[t].view(-1, 1, 1, 1)
    alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
    epsilon_theta = self.model(xt, t.view(-1))
    
    var_t = torch.gather(self.var, 0, t).view(-1, 1, 1, 1).to(device=self.device)
    mu_t = (1 / alpha_t.sqrt()) * (xt - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * epsilon_theta)
    eps = torch.randn(xt.shape, device=self.device)
    return mu_t + eps * torch.sqrt(var_t)

  # Sample cnt samples 
  def sample_x0(self, cnt):
    with torch.no_grad():
      x = torch.randn((cnt, 1, 28, 28), device=self.device)
      for t in reversed(range(self.T)):
        x = self.sample_p(x, x.new_full((cnt,), t, dtype=torch.long))
      normalized = x.clone()
      for i in range(len(normalized)):
          normalized[i] -= torch.min(normalized[i])
          normalized[i] *= 255 / torch.max(normalized[i])
      normalized = torch.round(normalized).to(dtype=torch.uint8)
      return normalized

    
def train_diffusion(dif_net, trainer, train_iter, valid_iter, num_epochs=20, device=get_device()):
  dif_net.to(device)
  
  for epoch in range(num_epochs):
    total_loss = 0
    imgs = dif_net.sample_x0(5)
    visualize_out(to_pil(imgs))
    for samples, _ in train_iter:
    
      samples = samples.to(device)

      dif_net.train()
      trainer.zero_grad()
      
      epsilon_hat, epsilon_0 = dif_net(samples)
      loss = torch.nn.functional.mse_loss(epsilon_hat, epsilon_0, reduction='mean')

      loss.backward()
      l = loss.cpu().detach().numpy()
      total_loss += l * len(samples)

      trainer.step()

    total_loss /= len(train_iter.dataset)
    print(f"Average loss for epoch {epoch} is: {total_loss}")

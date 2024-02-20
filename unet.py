import torch
from torch import nn


# A simplified version of the UNet used in DDPM with no attention layer
class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels, t_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    self.act1 = nn.ReLU()

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    self.act2 = nn.ReLU()

    self.t_mlp = nn.Linear(t_channels, out_channels)

  def forward(self, x, t):
    # Match number of channels from time embed to img
    t = self.t_mlp(t)
    
    x = self.act1(self.conv1(x))
    # (batch_size, out_channels, 1, 1)
    x = x + t[(..., ) + (None, ) * 2]
    x = self.conv2(x)

    return x

# Reduces dimensions by half
class UNetEncoder(nn.Module):
  def __init__(self, in_channels, out_channels, t_channels, num_blocks=2):
    super().__init__()
    
    self.convs = nn.ModuleList([ConvBlock(in_channels if i == 0 else out_channels, out_channels, t_channels) for i in range(num_blocks)])
    self.pool = nn.MaxPool2d((2, 2))

  def forward(self, x, t):
    for conv in self.convs:
      x = conv(x, t)

    skip = x
    x = self.pool(x)

    return x, skip
  
class UNetDecoder(nn.Module):
  def __init__(self, in_channels, out_channels, t_channels, num_blocks=2):
    super().__init__()
    # Doubles image dimensions
    self.convs = nn.ModuleList([ConvBlock(in_channels * 2 if i == 0 else out_channels, out_channels, t_channels) for i in range(num_blocks)])
    self.conv_up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0)

  def forward(self, x, t, skip):
    x = self.conv_up(x)
    x = torch.cat([x, skip], dim=1)
    for conv in self.convs:
      x = conv(x, t)


    return x

class TimeEmbedding(nn.Module):
  # T is max time step
  T = 2000

  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    x = torch.arange(self.T, dtype=torch.float32).reshape(-1, 1) / (torch.pow(10000, torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
    self.pos_embeds = nn.Parameter(torch.zeros((self.T, dim)), requires_grad=False)
    self.pos_embeds[:, 0::2] = torch.sin(x)
    self.pos_embeds[:, 1::2] = torch.cos(x)
    
  def forward(self, t):
    return self.pos_embeds[t].reshape(-1, self.dim)    
  
class UNetMiddle(nn.Module):
  def __init__(self, channels, t_channels):
    super().__init__()
    self.conv1 = ConvBlock(channels, channels, t_channels)
    self.act1 = nn.ReLU()
    self.conv2 = ConvBlock(channels, channels, t_channels)

  def forward(self, x, t):
    x = self.act1(self.conv1(x, t))
    x = self.conv2(x, t)
    return x


# UNet for a diffusion model
class UNet(nn.Module):
  def __init__(self, img_channels = 3, feat_channels = 64, feat_scales = [1, 2], t_channels = 256):
    super().__init__()
    # 1x1 conv to turn img into feat_channels and reverse
    self.to_feat = nn.Conv2d(img_channels, feat_channels, kernel_size=3, padding=1)
    self.to_img = nn.Conv2d(feat_channels, img_channels, kernel_size=3, padding=1)

    # Time embeddings
    self.time_emb = TimeEmbedding(t_channels)

    # Downscaling
    downscale = []
    in_channels = feat_channels
    for feat_scale in feat_scales:
      out_channels = feat_channels * feat_scale
      # Don't downscale on last
      downscale.append(UNetEncoder(in_channels, out_channels, t_channels))

      in_channels = out_channels
    self.downscale = nn.ModuleList(downscale)

    # Middle pass
    self.middle = UNetMiddle(out_channels, t_channels)

    # Upscaling
    upscale = []
    rev_feat_scales = list(reversed(feat_scales))[1:] + [1]
    in_channels = out_channels
    for feat_scale in rev_feat_scales:
      out_channels = feat_channels * feat_scale
      upscale.append(UNetDecoder(in_channels, out_channels, t_channels))

      in_channels = out_channels
    self.upscale = nn.ModuleList(upscale)



  def forward(self, x, t):
    # Get time embeddings
    t = self.time_emb(t)

    # Go to feat_channels
    x = self.to_feat(x)
    
    skips = []

    # Encoding/Contraction
    for down in self.downscale:
      x, skip = down(x, t)
      skips.append(skip)
    
    # Middle pass
    x = self.middle(x, t)

    # Decoding/Expansion
    for up, skip in zip(self.upscale, reversed(skips)):
      x = up(x, t, skip)

    # Go back to img_channels
    out = self.to_img(x)

    return out 




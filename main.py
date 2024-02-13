import torchvision
import torch
import vae
from torchvision import transforms
from matplotlib import pyplot as plt

def load_data():
  transform = transforms.Compose([transforms.ToTensor()])
  dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=transform)
  train_set, valid_set = torch.utils.data.random_split(dataset, [50000, 10000])
  train_iter = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
  valid_iter = torch.utils.data.DataLoader(valid_set, batch_size=256, shuffle=True)

  return train_iter, valid_iter

def get_device():
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, train_iter, valid_iter, latent_dim, num_epochs = 20, device=get_device()):
  for epoch in range(num_epochs):
    avg_loss = 0
    for examples, _ in train_iter:
      examples = examples.to(device)
      examples = examples.reshape(examples.shape[0], -1) # reshape into (batch_size, 28 * 28) 
      optimizer.zero_grad()
      net.train()

      # epsilon ~ N(0, I)
      epsilon = torch.randn((examples.shape[0], latent_dim)).to(device)
      x_hat, mu, log_var = net(examples, epsilon)

      elbo_loss = vae.ELBOLoss(mu, log_var, x_hat, examples)

      elbo_loss.backward()
      avg_loss += elbo_loss.cpu().detach().numpy()
      optimizer.step()
    avg_loss /= len(train_iter) * train_iter.batch_size
    print(f'Epoch {epoch} Avg Loss: {avg_loss}')

def visualize_out(net, count, latent_dim):
  z = torch.randn((count, latent_dim), device=get_device())
  out = net.decoder(z)

  imgs = out.reshape((count, 28, 28)).cpu().detach().numpy()

  num_col = 5
  num_row = count // num_col
  fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
  for i in range(count):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(imgs[i], cmap='gray')
  plt.tight_layout()
  plt.show()

def run_train(num_epochs = 20):
  train_iter, valid_iter = load_data()

  x_dim = 28 * 28
  # MNIST is 28 x 28
  encoder_hidden_layers = [400, 256]
  decoder_hidden_layers = [256, 400]
  latent_dim = 128

  encoder = vae.MLPEncoder(encoder_hidden_layers, latent_dim)
  decoder = vae.MLPDecoder(decoder_hidden_layers, x_dim)

  lr = 0.0001
  net = vae.VAE(encoder, decoder)

  device = get_device()
  net.to(device)

  optimizer = torch.optim.Adam(net.parameters(), lr)
  train(net, optimizer, train_iter, valid_iter, latent_dim, num_epochs)

  return net

if __name__ == "__main__":
  run_train()
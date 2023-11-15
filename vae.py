import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# dim of latent space
latent_dim = 20

# defining VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, latent_dim*2) # Predict both mean & var
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid() # Image must have pixel values between 0 and 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        # Encoder output
        #print("data: ", x.shape) # torch.Size([128, 1, 28, 28])
        x = x.view(-1, 784)
        #print("flatten data: ", x.shape) # torch.Size([128, 784])
        enc_output = self.encoder(x)
        #print("encoded data: ", enc_output.shape) # torch.Size([128, 40]) # 20: mean, 20: logvar

        # Extract mean & var
        mu, logvar = enc_output[:, :latent_dim], enc_output[:, latent_dim:]
        #print(f"latent's mu: {mu.shape}, latent's logvar: {logvar.shape}") # torch.Size([128, 20]), torch.Size([128, 20])

        # Sampling from latent space
        z = self.reparameterize(mu, logvar)
        #print("latent variable: ", z.shape) # torch.Size([128, 20]) # latent variable has 20 features

        # Decoder output
        dec_output = self.decoder(z)
        #print("dec_output: ", dec_output.shape) # torch.Size([128, 784])

        return dec_output, mu, logvar
    
# Define loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL Divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

# Data loading
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                                         batch_size=128, shuffle=True)

# Initialize Model & Setting optimizer
vae = VAE().to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# List for recording loss
train_loss_list = []

# List for recording Latent variable
latent_variables = []

# Train
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item() / len(data)}")

    # Record avg loss at the end of each epoch
    avg_loss = epoch_loss / len(train_loader.dataset)
    train_loss_list.append(avg_loss)
    print(f'Epoch {epoch}, Avg loss: {avg_loss}')

    # Record latent variable
    with torch.no_grad():
        z = torch.randn(len(train_loader.dataset), latent_dim)
        # print("z: ", z.shape) # torch.Size([60000, 20])
        latent_variables.append(vae.reparameterize(mu, logvar).cpu().numpy())
        #print("vae.reparameterize: ", vae.reparameterize(mu, logvar).cpu().numpy().shape) # (96, 20)

# Loss plot
plt.plot(range(1, num_epochs + 1), train_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Visualization of latent variable distribution
latent_variables = np.concatenate(latent_variables, axis=0)
#print("latent_variables: ", latent_variables.shape) # (960, 20)
plt.figure(figsize=(8, 6))
plt.scatter(latent_variables[:, 0], latent_variables[:, 1], c='b', alpha=0.5)
plt.title('Latent variable distribution')
plt.xlabel('Latent variable 1')
plt.ylabel('Latent variable 2')
plt.show()

# VAE Test by Test dataset
with torch.no_grad():
    z_sample = torch.randn(16, latent_dim).to(device)
    # print("z_sample: ", z_sample.shape) # torch.Size([16, 20])
    sample = vae.decoder(z_sample).view(16, 1, 28, 28).cpu()
    # print("sample: ", sample.shape) # torch.Size([16, 1, 28, 28])

# Visualize result
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i in range(16):
    axes[i // 4, i % 4].imshow(sample[i][0], cmap='gray')
    axes[i // 4, i % 4].axis('off')
plt.show()


# # Generate image by sampling from latent variable
# def generate_samples(vae, num_samples=16):
#     with torch.no_grad():
#         z_sample = torch.randn(num_samples, latent_dim).to(device)
#         print("z_sample: ", z_sample.shape)
#         generated_samples = vae.decoder(z_sample).view(num_samples, 1, 28, 28).cpu().numpy()
#         print("generated_samples: ", generated_samples.shape)
#     return generated_samples

# # Visualize generated samples from trained VAE model
# generated_samples = generate_samples(vae)

# # Visualize original MNIST image
# fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# for i in range(16):
#     axes[i // 4, i % 4].imshow(train_loader.dataset[i][0][0], cmap='gray')
#     axes[i // 4, i % 4].axis('off')
# plt.suptitle('Original MNIST Images', y=1.02)
# plt.show()

# # Visualize generated image
# fig, axes = plt.subplots(4, 4, figsize=(8, 8))
# generated_samples = generate_samples(vae)

# for i in range(16):
#     axes[i // 4, i % 4].imshow(generated_samples[i][0], cmap='gray')
#     axes[i // 4, i % 4].axis('off')
# plt.suptitle('Generated MNIST-like Images', y=1.02)
# plt.show()
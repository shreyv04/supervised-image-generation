# GANs using MNIST Dataset
# Import necessary libraries
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision 
import torchvision.datasets as datasets 
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 

#GAN Model
# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

# Define Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 128 * 7 * 7),
            nn.Unflatten(1, (128, 7, 7)), 
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output range [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

# GAN Training
# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
lr = 3e-4  # Learning rate
z_dim = 100  # Size of noise vector
batch_size = 64
num_epochs = 30

# Initialize models, loss, and optimizers
disc = Discriminator().to(device)
gen = Generator(z_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

# Use transforms for better normalization
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

# Load MNIST dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizers and Loss
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, z_dim).to(device)

        # Train Discriminator
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        # Print losses and display images
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28).cpu() # (batch size,channels,height,width)
                real = real.reshape(-1, 1, 28, 28).cpu()

                # Plot fake and real images
                fig, axes = plt.subplots(2, 10, figsize=(15, 6))
                for i in range(10):
                    axes[0, i].imshow(fake[i][0], cmap="gray")  # Fake images
                    axes[0, i].axis("off")
                    axes[1, i].imshow(real[i][0], cmap="gray")  # Real images
                    axes[1, i].axis("off")
                
                # Add row titles
                axes[0, 0].set_title("Generated Images", fontsize=14, pad=10)
                axes[1, 0].set_title("Real MNIST Images", fontsize=14, pad=10)

                plt.suptitle(f"Epoch {epoch + 1}")
                plt.tight_layout()
                plt.show()

# CGAN using MNIST Dataset
# Import necessary libraries 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
lr = 3e-4  # Learning rate
z_dim = 100  # Size of noise vector
image_channels = 1
image_size = 28
num_classes = 10
batch_size = 64
num_epochs = 30

# CGAN Model
# Define Generator
class Generator(nn.Module):
    def __init__(self, z_dim, image_channels, num_classes):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 128, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_embedding = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, label_embedding], dim=1)
        return self.model(x)

# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, image_channels, num_classes):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, image_size * image_size)
        self.model = nn.Sequential(
            nn.Conv2d(image_channels + 1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_embedding = self.label_embedding(labels).view(labels.size(0), 1, image_size, image_size)
        x = torch.cat([x, label_embedding], dim=1)
        return self.model(x)

# CGAN Training
# Initialize models
gen = Generator(z_dim, image_channels, num_classes).to(device)
disc = Discriminator(image_channels, num_classes).to(device)

# Fixed noise and labels for visualization
fixed_noise = torch.randn((num_classes, z_dim, 1, 1)).to(device)
fixed_labels = torch.arange(0, num_classes).to(device)

# Transforms for the dataset
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Optimizers and Loss
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    for batch_idx, (real, labels) in enumerate(loader):
        real = real.to(device)
        labels = labels % num_classes
        labels = labels.to(device)
        batch_size = real.size(0)

        # Train Discriminator
        noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
        fake_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        fake = gen(noise, fake_labels)
        disc_real = disc(real, labels).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach(), fake_labels).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        # Train Generator
        output = disc(fake, fake_labels).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

    # Display progress for each epoch
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}")
    
    # Generate and display images at the end of each epoch
    with torch.no_grad():
        fake_images = gen(fixed_noise, fixed_labels).detach().cpu()
        real_images = real[:num_classes].cpu()
    
    # Plot generated and real images
    fig, axes = plt.subplots(2, num_classes, figsize=(15, 6))
    for i in range(num_classes):
        axes[0, i].imshow(fake_images[i][0], cmap="gray")  
        axes[0, i].set_title(f"Generated: {i}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(real_images[i][0], cmap="gray")  
        axes[1, i].axis("off")
    
    # Add titles for rows
    fig.text(0.5, 0.9, "Generated Images", ha='center', fontsize=12, weight="bold")
    fig.text(0.5, 0.48, "Real MNIST Images", ha='center', fontsize=12, weight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# GAN using CelebA Dataset
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# GAN Model
# Define Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1)
        )

    def forward(self, x):
        return self.disc(x)

# Define Generator
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256 * 4 * 4),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# GAN Training
if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    lr = 2e-4
    z_dim = 100
    batch_size = 256
    num_epochs = 5
    image_size = 64

    # Initialize models
    disc = Discriminator().to(device)
    gen = Generator(z_dim).to(device)
    fixed_noise = torch.randn((batch_size, z_dim)).to(device)

    # Data transforms for CelebA
    transform_pipeline = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CelebA dataset
    dataset = datasets.CelebA(root="dataset/", split="train", transform=transform_pipeline, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Optimizers and Loss
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    # Mixed Precision Training
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    # Training Loop
    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(device)
            batch_size = real.size(0)
            noise = torch.randn(batch_size, z_dim).to(device)

            # Train Discriminator
            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
                fake = gen(noise)
                disc_real = disc(real).view(-1)
                lossD_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(fake.detach()).view(-1)
                lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                lossD = (lossD_real + lossD_fake) / 2

            disc.zero_grad()
            scaler.scale(lossD).backward()
            scaler.step(opt_disc)
            scaler.update()

            # Train Generator
            with torch.autocast(device_type=device, dtype=torch.float16 if device == "cuda" else torch.float32):
                output = disc(fake).view(-1)
                lossG = criterion(output, torch.ones_like(output))

            gen.zero_grad()
            scaler.scale(lossG).backward()
            scaler.step(opt_gen)
            scaler.update()

            # Print losses and display images
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} "
                    f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
                )

                with torch.no_grad():
                    fake = gen(fixed_noise).reshape(-1, 3, image_size, image_size).cpu()
                    real = real[:10].cpu()

                    # Plot fake and real images
                    fig, axes = plt.subplots(2, 10, figsize=(15, 6))  # 2 rows, 10 columns
                    for i in range(10):
                        axes[0, i].imshow(fake[i].permute(1, 2, 0) * 0.5 + 0.5)  # Fake images
                        axes[0, i].axis("off")
                        axes[1, i].imshow(real[i].permute(1, 2, 0) * 0.5 + 0.5)  # Real images
                        axes[1, i].axis("off")

                    # Add row titles
                    axes[0, 0].set_title("Generated Images", fontsize=14, pad=10)
                    axes[1, 0].set_title("Real CelebA Images", fontsize=14, pad=10)

                    plt.suptitle(f"Epoch {epoch + 1}")
                    plt.tight_layout()
                    plt.show()
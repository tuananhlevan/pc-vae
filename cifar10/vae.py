import torch
import torch.nn as nn

class ResNetVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # ENCODER: 32x32x3 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 4x4
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Latent Projections
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # DECODER: 128 -> 4x4 -> 8x8 -> 16x16 -> 32x32x3
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Scale to [0, 1] for CIFAR-10 images
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        
        print(f"LogVar Min: {logvar.min().item():.2f} | LogVar Max: {logvar.max().item():.2f}")
        logvar = torch.clamp(logvar, min=-10., max=10.)
        return mu, logvar

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, 128, 4, 4) # Reshape back to feature map
        return self.decoder(h)
import torch
import torch.nn as nn
import pyjuice as juice
import pyjuice.nodes.distributions as dists

class ResNetAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # ENCODER: 32x32x3 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim)
        )
        
        # DECODER: Upsampling blocks to prevent checkerboard artifacts
        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z = self.encoder(x)
        
        # Gatekeeper: Restrict latents to [-5, 5] to protect the PC from infinite variances
        z = 5.0 * torch.tanh(z / 5.0) 
        
        h = self.decoder_input(z)
        h = h.view(-1, 128, 4, 4)
        x_recon = self.decoder(h)
        return x_recon, z

def build_hclt_prior(latent_data, latent_dim, device):
    """
    Constructs the Hierarchical Chow-Liu Tree Probabilistic Circuit.
    Requires a subset of data to calculate mutual information for the tree structure.
    """
    ns = juice.structures.HCLT(
        latent_data.float().to(device), 
        num_bins=32,       
        sigma=0.5 / 32,    
        num_latents=latent_dim,   
        chunk_size=32,
        # Explicitly instruct PyJuice to use continuous Gaussians, not Categorical pixels
        input_dist=dists.Gaussian(mu=0.0, sigma=1.0, min_sigma=0.05) 
    )
    
    # Perturb weights so EM has room to optimize
    ns.init_parameters(perturbation=2.0)
    
    # Compile to GPU
    pc_prior = juice.TensorCircuit(ns).to(device)
    return pc_prior
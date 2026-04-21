import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PCPrior(nn.Module):
    """
    The simplest Probabilistic Circuit: A Gaussian Mixture Model (Depth-2 PC).
    Root: Sum Node (Mixture Weights)
    Leaves: Product Nodes of independent Univariate Gaussians
    """
    def __init__(self, num_components, latent_dim):
        super().__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim
        
        # Sum node weights (log-space for stability)
        self.mixture_logits = nn.Parameter(torch.zeros(num_components))
        
        # Leaf nodes: Independent learnable Gaussians
        self.means = nn.Parameter(torch.randn(num_components, latent_dim))
        self.log_vars = nn.Parameter(torch.zeros(num_components, latent_dim))

    def exact_log_prob(self, z):
        # z shape: [Batch, Latent_Dim]
        
        # Expand z to match components: [Batch, Components, Latent_Dim]
        z_expanded = z.unsqueeze(1).expand(-1, self.num_components, -1)
        
        # Expand params: [1, Components, Latent_Dim]
        means = self.means.unsqueeze(0)
        log_vars = self.log_vars.unsqueeze(0)
        
        # Calculate log probability of z under each independent leaf Gaussian
        # Formula: -0.5 * [log(2*pi) + log_var + (z - mean)^2 / exp(log_var)]
        log_p_leaves = -0.5 * (math.log(2 * math.pi) + log_vars + ((z_expanded - means) ** 2) * torch.exp(-log_vars))
        
        # Product nodes (Sum in log-space across the independent latent dimensions)
        # Shape becomes: [Batch, Components]
        log_p_product = log_p_leaves.sum(dim=-1)
        
        # Sum node (LogSumExp with learnable mixture weights)
        log_weights = F.log_softmax(self.mixture_logits, dim=0).unsqueeze(0)
        log_p_sum = torch.logsumexp(log_p_product + log_weights, dim=1)
        
        return log_p_sum # The exact log p(z) calculation

class PC_VAE(nn.Module):
    def __init__(self, latent_dim=16, num_components=10):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder (Approximates posterior q(z|x))
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_logvar = nn.Linear(200, latent_dim)
        
        # Exact PC Prior p(z) - 10 clusters to roughly match 10 MNIST digits
        self.pc_prior = PCPrior(num_components, latent_dim)
        
        # Decoder p(x|z)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid() # Output pixels in range [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

def pc_vae_loss(x_recon, x, mu, logvar, z, model):
    """
    Computes the Monte Carlo ELBO for the PC-VAE.
    """
    # 1. Reconstruction Loss: -log p(x|z)
    recon_loss = F.binary_cross_entropy(x_recon, x.view(-1, 28 * 28), reduction='none').sum(dim=-1)
    
    # 2. Exact Prior Log-Likelihood: log p_{pc}(z)
    log_p_z = model.pc_prior.exact_log_prob(z)
    
    # 3. Encoder Entropy: log q(z|x)
    # Probability of the sampled z under the encoder's predicted Gaussian
    log_q_z = -0.5 * (math.log(2 * math.pi) + logvar + ((z - mu) ** 2) * torch.exp(-logvar)).sum(dim=-1)
    
    # ELBO = E_q [ log p(x|z) + log p(z) - log q(z|x) ]
    # Since PyTorch minimizes loss, we return the negative ELBO
    elbo = -recon_loss + log_p_z - log_q_z
    
    return -elbo.mean()


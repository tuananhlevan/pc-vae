import torch
import math
import torch.nn.functional as F

def train_step(x, vae_model, pc_prior, optimizer_vae, optimizer_pc):
    optimizer_vae.zero_grad()
    optimizer_pc.zero_grad()
    
    # 1. Encode
    mu, logvar = vae_model.encode(x)
    
    # 2. Sample (Reparameterization Trick)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    z = mu + eps * std
    
    # 3. Decode & Reconstruction Loss
    x_recon = vae_model.decode(z)
    recon_loss = F.mse_loss(x_recon, x, reduction='none').view(x.size(0), -1).sum(dim=-1)
    
    # 4. EXACT Prior Likelihood via PyJuice
    # PyJuice expects data in shape [Batch, Num_Vars]
    log_p_z = pc_prior(z)
    
    # 5. Encoder Entropy
    # log_q_z = -0.5 * (math.log(2 * math.pi) + logvar + ((z - mu) ** 2) / torch.exp(logvar)).sum(dim=-1)
    log_q_z = -0.5 * (math.log(2 * math.pi) + logvar + (eps ** 2)).sum(dim=-1)
    
    # 6. Optimize ELBO
    beta = 0.03
    elbo = -recon_loss + beta * (log_p_z - log_q_z)
    loss = -elbo.mean()
    
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(pc_prior.parameters(), max_norm=5.0)
    
    # Step both optimizers
    optimizer_vae.step()
    optimizer_pc.step()
    
    return loss.item()
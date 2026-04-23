import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import lpips 

from model import VQVAE, build_discrete_hclt_prior, PatchGANDiscriminator

torch.manual_seed(42)

# --- Hinge Losses ---
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    return 0.5 * (loss_real + loss_fake)

def hinge_g_loss(logits_fake):
    return -torch.mean(logits_fake)

def calculate_adaptive_weight(loss_l1, loss_g, last_layer_weight):
    # Get the gradients of the L1 and GAN loss with respect to the last layer
    l1_grads = torch.autograd.grad(loss_l1, last_layer_weight, retain_graph=True)[0]
    g_grads = torch.autograd.grad(loss_g, last_layer_weight, retain_graph=True)[0]
    
    # Calculate the norm of the gradients
    d_weight = torch.norm(l1_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    return d_weight

def train_vqvae(device, batch_size=128, epochs=500, disc_start_epoch=50):
    print("\n--- PHASE 1: Training VQ-VAE (with Discriminator) ---")
    model = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    discriminator = PatchGANDiscriminator().to(device) # NEW: Initialize Discriminator
    
    # Keep the Generator fast so it can aggressively search for better textures
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))

    # Handicap the Discriminator to be 4x to 5x slower
    opt_disc = optim.Adam(discriminator.parameters(), lr=5e-4, betas=(0.5, 0.9))
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_d_loss = 0 # NEW: Track D Loss
        
        for x, _ in loader:
            x = x.to(device)
            
            # ==============================================================
            # 1. Train Generator (VQ-VAE)
            # ==============================================================
            optimizer.zero_grad()
            
            x_recon, vq_loss, _ = model(x)
            
            loss_l1 = F.l1_loss(x_recon, x)
            loss_lpips = loss_fn_vgg(x_recon * 2.0 - 1.0, x * 2.0 - 1.0).mean()
            
            # Adversarial Loss (Activates after warmup)
            if epoch >= disc_start_epoch:
                logits_fake = discriminator(x_recon)
                loss_g = hinge_g_loss(logits_fake)
                
                last_layer = model.decoder[-2].weight 
                adaptive_gan_weight = calculate_adaptive_weight(loss_l1, loss_g, last_layer)
                
                loss = loss_l1 + (0.25 * loss_lpips) + vq_loss + (adaptive_gan_weight * loss_g)
            else:
                loss_g = torch.tensor(0.0).to(device)
                loss = loss_l1 + (0.25 * loss_lpips) + vq_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # ==============================================================
            # 2. Train Discriminator
            # ==============================================================
            if epoch >= disc_start_epoch:
                opt_disc.zero_grad()
                
                # Detach x_recon so gradients don't flow back into the Generator
                logits_real = discriminator(x.detach())
                logits_fake = discriminator(x_recon.detach())
                
                d_loss = hinge_d_loss(logits_real, logits_fake)
                d_loss.backward()
                opt_disc.step()
                total_d_loss += d_loss.item()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Calculate average losses for printing
        avg_g_loss = total_loss / len(loader)
        avg_d_loss = total_d_loss / len(loader) if epoch >= disc_start_epoch else 0.0
        
        print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.6f} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

    torch.save(model.state_dict(), 'checkpoints/vqvae_stage1.pth')
    torch.save(discriminator.state_dict(), 'checkpoints/discriminator_stage1.pth') # Optional: save discriminator
    print("Phase 1 Complete. VQ-VAE Saved.")

def extract_discrete_latents(device):
    print("\n--- PHASE 2: Extracting Integer Tokens ---")
    model = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    model.load_state_dict(torch.load('checkpoints/vqvae_stage1.pth', map_location=device, weights_only=True))
    model.eval()

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    all_indices = []
    with torch.no_grad():
        for x, _ in loader:
            _, _, indices = model(x.to(device))
            all_indices.append(indices.cpu())

    # Shape will be [50000, 64] where every value is an integer from 0 to 127
    latents = torch.cat(all_indices, dim=0)
    
    torch.save({'latents': latents}, 'checkpoints/cifar10_discrete_latents.pt')
    print(f"Active Tokens Used: {len(torch.unique(latents))}/128")
    print(f"Phase 2 Complete. {latents.shape[0]} Discrete Maps Saved.")

def train_discrete_pc(device, epochs=100):
    print("\n--- PHASE 3: Training Exact Categorical HCLT ---")
    data = torch.load('checkpoints/cifar10_discrete_latents.pt', map_location='cpu')
    latents = data['latents'].view(-1, 64) # Shape: [50000, 64]
    
    loader = DataLoader(TensorDataset(latents), batch_size=512, shuffle=True, drop_last=True)
    
    print("Building Discrete HCLT Structure...")
    # Pass 10k samples to build the MI dependency tree
    pc_prior = build_discrete_hclt_prior(latents, num_cats=128, device=device)
    
    step_size = 0.05
    for epoch in range(epochs):
        total_ll = 0.0
        batches = 0
        
        for z_batch, in loader:
            # PyJuice Categorical expects integer types for lookup
            z_batch = z_batch.long().to(device)
            
            pc_prior.zero_param_flows()
            
            lls = pc_prior(z_batch) 
            total_ll += lls.mean().item()
            
            pc_prior.backward(z_batch, compute_param_flows=True)
            pc_prior.mini_batch_em(step_size=step_size, pseudocount=0.1)
            batches += 1
            
        print(f"Epoch [{epoch+1}/{epochs}] | Exact Log-Likelihood: {total_ll/batches:.4f}")

    torch.save(pc_prior.state_dict(), 'checkpoints/vq_hclt_pc.pth')
    print("Phase 3 Complete. Discrete Prior Saved.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs('checkpoints', exist_ok=True)
    
    if not os.path.exists('checkpoints/vqvae_stage1.pth'):
        train_vqvae(device)
        
    if not os.path.exists('checkpoints/cifar10_discrete_latents.pt'):
        extract_discrete_latents(device)
        
    if not os.path.exists('checkpoints/vq_hclt_pc.pth'):
        train_discrete_pc(device)
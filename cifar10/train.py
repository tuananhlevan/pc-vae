import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import lpips 

from model import ResNetAE, build_hclt_prior

def train_vae(device, latent_dim, batch_size=128, epochs=40):
    print("\n--- PHASE 1: Training Canvas (VAE) ---")
    model = ResNetAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # LPIPS Perceptual Loss to force sharp textures
    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            x = x.to(device)
            optimizer.zero_grad()
            
            x_recon, _ = model(x)
            
            loss_l1 = F.l1_loss(x_recon, x)
            
            # Scale outputs to [-1, 1] for LPIPS compatibility
            loss_lpips = loss_fn_vgg(x_recon * 2.0 - 1.0, x * 2.0 - 1.0).mean()
            
            loss = loss_l1 + (0.5 * loss_lpips)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] | VAE Combined Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), 'checkpoints/clean_ae_stage1.pth')
    print("Phase 1 Complete. VAE Saved.")

def extract_latents(device, latent_dim):
    print("\n--- PHASE 2: Extracting Static Latent Dataset ---")
    model = ResNetAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load('checkpoints/clean_ae_stage1.pth', map_location=device, weights_only=True))
    model.eval() # Freeze the network

    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    all_latents = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            _, z = model(x.to(device))
            all_latents.append(z.cpu())
            all_labels.append(y.cpu())

    latents = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    torch.save({'latents': latents, 'labels': labels}, 'checkpoints/cifar10_static_latents.pt')
    print(f"Phase 2 Complete. {latents.shape[0]} Latents Saved.")

def train_pc(device, latent_dim, epochs=30):
    print("\n--- PHASE 3: Training Painter (HCLT Prior) ---")
    data = torch.load('checkpoints/cifar10_static_latents.pt', map_location='cpu')
    latents = data['latents']
    
    # drop_last=True prevents PyJuice batch mismatch errors
    loader = DataLoader(TensorDataset(latents), batch_size=512, shuffle=True, drop_last=True)
    
    print("Building HCLT Structure...")
    pc_prior = build_hclt_prior(latents[:10000], latent_dim, device)
    
    step_size = 0.05
    for epoch in range(epochs):
        total_ll = 0.0
        batches = 0
        
        for z_batch, in loader:
            z_batch = z_batch.to(device)
            
            # CRITICAL: Flush the EM buffers so it doesn't freeze!
            pc_prior.zero_param_flows()
            
            # Forward (Expectation)
            lls = pc_prior(z_batch) 
            total_ll += lls.mean().item()
            
            # Backward (Flow Accumulation)
            pc_prior.backward(z_batch, compute_param_flows=True)
            
            # Update (Maximization)
            pc_prior.mini_batch_em(step_size=step_size, pseudocount=0.1)
            batches += 1
            
        print(f"Epoch [{epoch+1}/{epochs}] | Exact Log-Likelihood: {total_ll/batches:.4f}")

    torch.save(pc_prior.state_dict(), 'checkpoints/hclt_pc_weights.pth')
    print("Phase 3 Complete. PC Prior Saved.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    os.makedirs('checkpoints', exist_ok=True)
    
    # Auto-execute pipeline sequentially
    if not os.path.exists('checkpoints/clean_ae_stage1.pth'):
        train_vae(device, latent_dim)
        
    if not os.path.exists('checkpoints/cifar10_static_latents.pt'):
        extract_latents(device, latent_dim)
        
    if not os.path.exists('checkpoints/hclt_pc_weights.pth'):
        train_pc(device, latent_dim)
        
    print("\nAll Training Phases Complete! You can now run generate.py")
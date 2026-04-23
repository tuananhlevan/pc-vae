import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from model import VQVAE

# --- ADDED: Visualization Function ---
def visualize_reconstructions(real_imgs, recon_imgs, num_images=8, filename="assets/reconstruction_comparison.png"):
    """Saves and displays a grid comparing real and reconstructed images."""
    print("Generating visualization...")
    # Cap the number of images to display
    num_images = min(num_images, real_imgs.size(0))
    
    real_imgs = real_imgs[:num_images].cpu()
    recon_imgs = recon_imgs[:num_images].cpu()
    
    # Ensure reconstructed pixel values are valid [0, 1] for matplotlib
    recon_imgs = torch.clamp(recon_imgs, 0.0, 1.0)

    # Stack them: Top row will be Real, Bottom row will be Reconstructed
    comparison = torch.cat([real_imgs, recon_imgs])
    
    # Create a single grid image using torchvision
    grid = make_grid(comparison, nrow=num_images, padding=2, normalize=False)
    
    # Convert to numpy and transpose to (Height, Width, Channels) for matplotlib
    grid_np = np.transpose(grid.numpy(), (1, 2, 0))
    
    plt.figure(figsize=(num_images * 2, 4))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title('Top Row: Original Real Images  |  Bottom Row: VQ-VAE Reconstructions', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved visual comparison to '{filename}'")
# -------------------------------------

def evaluate_reconstruction_fid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_eval_samples = 10000 
    batch_size = 128
    
    print(f"Evaluating VQ-VAE Absolute Ceiling on {device}...")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load Phase 1 Model
    vae_model = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    vae_model.load_state_dict(torch.load('checkpoints/vqvae_stage1.pth', map_location=device, weights_only=True))
    vae_model.eval()

    processed = 0
    viz_real, viz_recon = None, None # To store images for visualization

    with torch.no_grad():
        for i, (x, _) in enumerate(tqdm(loader, desc="Processing Images")):
            if processed >= num_eval_samples:
                break
            
            x = x.to(device)
            
            # 1. Update FID with REAL images
            fid.update(x, real=True)
            
            # 2. Reconstruct the exact same images
            x_recon, _, _ = vae_model(x)
            
            # 3. Update FID with RECONSTRUCTED images
            fid.update(x_recon, real=False)
            
            # --- ADDED: Save the very first batch for our visualization ---
            if i == 0:
                viz_real = x.clone()
                viz_recon = x_recon.clone()
            # -------------------------------------------------------------

            processed += x.size(0)

    # Run the visualization before printing the final score
    if viz_real is not None and viz_recon is not None:
        visualize_reconstructions(viz_real, viz_recon, num_images=8)

    fid_score = fid.compute()
    print("-" * 40)
    print(f"RECONSTRUCTION FID SCORE (The Ceiling): {fid_score.item():.2f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_reconstruction_fid()
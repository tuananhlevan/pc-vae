import torch
import pyjuice as juice
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

from model import VQVAE, build_discrete_hclt_prior

def generate_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing VQ-Generator on {device}...")

    # 1. Load VQ-VAE
    vae_model = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    vae_model.load_state_dict(torch.load('checkpoints/vqvae_stage1.pth', map_location=device, weights_only=True))
    vae_model.eval()

    # 2. Rebuild & Load Discrete HCLT Structure
    data = torch.load('checkpoints/cifar10_discrete_latents.pt', map_location='cpu')
    pc_prior = build_discrete_hclt_prior(data['latents'].view(-1, 64), num_cats=128, device=device)
    pc_prior.load_state_dict(torch.load('checkpoints/vq_hclt_pc.pth', map_location=device, weights_only=True))

    num_samples = 64
    print("Sampling Discrete Tokens from PC Prior...")
    with torch.no_grad():
        # Sample exact integer tokens from the PyJuice HCLT (Shape: [64, 64])
        z_indices = juice.queries.sample(pc_prior, num_samples=num_samples).to(device)
        
        # Look up the continuous embeddings for those specific tokens
        z_q = vae_model.vq.embedding[z_indices.long()]
        
        # Reshape [64 samples, 64 tokens, 64 channels] -> [64, 64, 8, 8]
        z_q = z_q.view(-1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
        
        # Decode the perfect patches into an image
        generated_imgs = vae_model.decoder(z_q).cpu()

    # 5. Plot
    grid = make_grid(generated_imgs, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title("Discrete VQ-HCLT Generations")
    plt.tight_layout()
    
    os.makedirs('assets', exist_ok=True)
    plt.savefig('assets/cifar10_vq_pc_samples.png', bbox_inches='tight', pad_inches=0.1)
    print("Success! Open 'assets/cifar10_vq_pc_samples.png'")

if __name__ == "__main__":
    generate_images()
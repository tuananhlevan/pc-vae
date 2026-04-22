import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from model import VQVAE

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
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Processing Images"):
            if processed >= num_eval_samples:
                break
            
            x = x.to(device)
            
            # 1. Update FID with REAL images
            fid.update(x, real=True)
            
            # 2. Reconstruct the exact same images
            x_recon, _, _ = vae_model(x)
            
            # 3. Update FID with RECONSTRUCTED images
            fid.update(x_recon, real=False)
            
            processed += x.size(0)

    fid_score = fid.compute()
    print("-" * 40)
    print(f"RECONSTRUCTION FID SCORE (The Ceiling): {fid_score.item():.2f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_reconstruction_fid()
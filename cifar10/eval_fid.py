import torch
import pyjuice as juice
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

# Import the new VQ architectures
from model import VQVAE, build_discrete_hclt_prior

def evaluate_vq_fid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_eval_samples = 10000 
    batch_size = 128
    
    print(f"Initializing VQ-VAE FID Evaluator on {device}...")

    # 1. Load TorchMetrics FID
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # 2. Process Real CIFAR-10 Images
    print("Extracting Inception features from Real CIFAR-10...")
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    real_processed = 0
    with torch.no_grad():
        for x, _ in tqdm(real_loader, desc="Real Images"):
            if real_processed >= num_eval_samples:
                break
            
            x = x.to(device)
            fid.update(x, real=True)
            real_processed += x.size(0)

    # 3. Load the Discrete VQ-VAE Pipeline
    print("Loading VQ-VAE Generator...")
    vae_model = VQVAE(num_embeddings=128, embedding_dim=64).to(device)
    vae_model.load_state_dict(torch.load('checkpoints/vqvae_stage1.pth', map_location=device, weights_only=True))
    vae_model.eval()

    # Load and reshape the static discrete latents for structural building
    data = torch.load('checkpoints/cifar10_discrete_latents.pt', map_location='cpu')
    latents = data['latents'].view(-1, 64) # Ensure it is [50000, 64]
    
    pc_prior = build_discrete_hclt_prior(latents, num_cats=128, device=device)
    pc_prior.load_state_dict(torch.load('checkpoints/vq_hclt_pc.pth', map_location=device, weights_only=True))

    # 4. Process Generated Images
    print("Extracting Inception features from VQ Generated Images...")
    generated_processed = 0
    with torch.no_grad():
        with tqdm(total=num_eval_samples, desc="Generated Images") as pbar:
            while generated_processed < num_eval_samples:
                current_batch = min(batch_size, num_eval_samples - generated_processed)
                
                # A. Sample exact integer tokens from the PyJuice HCLT (Shape: [Batch, 64])
                z_indices = juice.queries.sample(pc_prior, num_samples=current_batch).to(device)
                
                # B. Look up the continuous embeddings from the codebook
                z_q = vae_model.vq.embedding[z_indices.long()] 
                
                # C. Reshape to feature map [Batch, 64 Channels, 8 Height, 8 Width]
                z_q = z_q.view(-1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
                
                # D. Decode the patches into an image
                generated_imgs = vae_model.decoder(z_q)
                
                # Update FID metric
                fid.update(generated_imgs, real=False)
                
                generated_processed += current_batch
                pbar.update(current_batch)

    # 5. Compute Final Score
    print("Calculating final Frechet Distance (this might take a moment)...")
    fid_score = fid.compute()
    
    print("-" * 40)
    print(f"FINAL VQ-VAE FID SCORE: {fid_score.item():.2f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_vq_fid()
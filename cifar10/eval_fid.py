import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from model import ResNetAE, build_hclt_prior

def evaluate_fid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    num_eval_samples = 10000 # Standard FID uses 10k or 50k samples
    batch_size = 128
    
    print(f"Initializing FID Evaluator on {device}...")

    # 1. Load the TorchMetrics FID module
    # feature=2048 is the standard InceptionV3 pool3 layer used in the official paper
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # 2. Process Real CIFAR-10 Images
    print("Extracting Inception features from Real CIFAR-10...")
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_processed = 0
    with torch.no_grad():
        for x, _ in tqdm(real_loader, desc="Real Images"):
            if real_processed >= num_eval_samples:
                break
            
            # torchmetrics expects images in [0, 1] if normalize=True
            x = x.to(device)
            fid.update(x, real=True)
            real_processed += x.size(0)

    # 3. Load the PC-VAE Pipeline
    print("Loading PC-VAE Generator...")
    vae_model = ResNetAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load('checkpoints/clean_ae_stage1.pth', map_location=device, weights_only=True))
    vae_model.eval()

    data = torch.load('checkpoints/cifar10_static_latents.pt', map_location='cpu')
    pc_prior = build_hclt_prior(data['latents'][:10000], latent_dim, device)
    pc_prior.load_state_dict(torch.load('checkpoints/hclt_pc_weights.pth', map_location=device, weights_only=True))

    # Apply Temperature Sampling
    temperature = 0.8
    with torch.no_grad():
        for layer in pc_prior.input_layer_group:
            if hasattr(layer, 'params') and layer.params is not None:
                layer.params[1::2] *= temperature

    # 4. Process Generated Images
    print("Extracting Inception features from Generated Images...")
    generated_processed = 0
    with torch.no_grad():
        with tqdm(total=num_eval_samples, desc="Generated Images") as pbar:
            while generated_processed < num_eval_samples:
                current_batch = min(batch_size, num_eval_samples - generated_processed)
                
                # Sample latents from PyJuice
                z_samples = juice.queries.sample(pc_prior, num_samples=current_batch).to(device)
                
                # Decode
                h = vae_model.decoder_input(z_samples)
                h = h.view(-1, 128, 4, 4)
                generated_imgs = vae_model.decoder(h)
                
                # Update FID metric
                fid.update(generated_imgs, real=False)
                
                generated_processed += current_batch
                pbar.update(current_batch)

    # 5. Compute Final Score
    print("Calculating final Frechet Distance (this might take a moment)...")
    fid_score = fid.compute()
    
    print("-" * 40)
    print(f"FINAL FID SCORE: {fid_score.item():.2f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_fid()
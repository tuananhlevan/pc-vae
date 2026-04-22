import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from model import PC_VAE

def sample_from_pc_prior(pc_prior, num_samples, device):
    """
    Samples exact vectors from the trained Probabilistic Circuit.
    """
    with torch.no_grad():
        cluster_probs = F.softmax(pc_prior.mixture_logits, dim=0)
        component_indices = torch.multinomial(cluster_probs, num_samples, replacement=True)
        
        selected_means = pc_prior.means[component_indices]
        selected_log_vars = pc_prior.log_vars[component_indices]
        
        std = torch.exp(0.5 * selected_log_vars)
        eps = torch.randn_like(std)
        z_samples = selected_means + eps * std
        
        return z_samples

def evaluate_mnist_fid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_eval_samples = 10000 
    batch_size = 128
    
    print(f"Initializing MNIST FID Evaluator on {device}...")

    # 1. Load the TorchMetrics FID module
    # normalize=True expects inputs in the [0.0, 1.0] range
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # 2. Process Real MNIST Images
    print("Extracting Inception features from Real MNIST...")
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    real_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    real_processed = 0
    with torch.no_grad():
        for x, _ in tqdm(real_loader, desc="Real Images"):
            if real_processed >= num_eval_samples:
                break
            
            x = x.to(device)
            # CRITICAL FIX: Duplicate the 1 grayscale channel into 3 RGB channels
            x_3c = x.repeat(1, 3, 1, 1) 
            
            fid.update(x_3c, real=True)
            real_processed += x.size(0)

    # 3. Load the PC-VAE Model
    print("Loading PC-VAE Generator...")
    model = PC_VAE(latent_dim=16, num_components=10).to(device)
    model.load_state_dict(torch.load('checkpoints/pc_vae_mnist.pth', map_location=device, weights_only=True))
    model.eval()

    # 4. Process Generated Images
    print("Extracting Inception features from Generated Images...")
    generated_processed = 0
    with torch.no_grad():
        with tqdm(total=num_eval_samples, desc="Generated Images") as pbar:
            while generated_processed < num_eval_samples:
                current_batch = min(batch_size, num_eval_samples - generated_processed)
                
                # Sample latents from the 10-component PyTorch PC
                z_samples = sample_from_pc_prior(model.pc_prior, current_batch, device)
                
                # Decode the samples back into images
                generated_imgs = model.decoder(z_samples)
                
                # Ensure the shape is [Batch, 1, 28, 28] before repeating
                generated_imgs = generated_imgs.view(-1, 1, 28, 28)
                
                # Convert generated 1-channel to 3-channel
                generated_3c = generated_imgs.repeat(1, 3, 1, 1)
                
                # Update FID metric
                fid.update(generated_3c, real=False)
                
                generated_processed += current_batch
                pbar.update(current_batch)

    # 5. Compute Final Score
    print("Calculating final Frechet Distance...")
    fid_score = fid.compute()
    
    print("-" * 40)
    print(f"FINAL MNIST FID SCORE: {fid_score.item():.2f}")
    print("-" * 40)

if __name__ == "__main__":
    evaluate_mnist_fid()
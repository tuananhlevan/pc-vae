import torch
import pyjuice as pj
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cifar10.vae import ResNetVAE
from cifar10.pc_prior import build_pyjuice_prior

def manual_pc_sample(pc_prior, num_samples, latent_dim, num_clusters, device):
    """
    Manually samples a 1-Layer PC (Sum -> Prod -> Normal Leaves) 
    by extracting the underlying PyTorch tensors.
    """
    with torch.no_grad():
        # 1. Grab the Root Sum Node Weights (The Mixture Probabilities)
        # In a 1-layer PC, the root node parameters dictate which cluster to pick
        # Get the raw parameters and apply softmax to get valid probabilities
        sum_params = pc_prior.inner_layer_params[0] # Exact index might vary slightly by version
        cluster_probs = torch.softmax(sum_params, dim=-1)
        
        # Sample which "island" (cluster) each of the 64 images belongs to
        chosen_clusters = torch.multinomial(cluster_probs, num_samples, replacement=True)
        
        # 2. Extract the Leaf Node (Gaussian) Parameters
        # PyJuice stores leaf parameters in a flat tensor. 
        # We extract them and reshape them into [Latent_Dim, Num_Clusters]
        # (Assuming dists.Normal stores mu and sigma contiguously)
        leaf_params = pc_prior.input_layer_param
        
        # Note: PyJuice internal parameter shapes can be tricky. 
        # Usually, they are stacked as [mu, sigma] for each variable and cluster.
        means = leaf_params[:, 0].view(latent_dim, num_clusters)
        sigmas = leaf_params[:, 1].view(latent_dim, num_clusters)
        
        # 3. Construct the exact samples
        z_samples = torch.zeros(num_samples, latent_dim, device=device)
        
        for i in range(num_samples):
            c_idx = chosen_clusters[i]
            # Grab the specific 128-dimensional mean and std for the chosen cluster
            mu = means[:, c_idx]
            sigma = sigmas[:, c_idx]
            
            # Reparameterization sample
            eps = torch.randn(latent_dim, device=device)
            z_samples[i] = mu + eps * sigma
            
        return z_samples

def generate_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 128
    
    # 1. Initialize Architectures
    vae_model = ResNetVAE(latent_dim=latent_dim).to(device)
    pc_prior = build_pyjuice_prior(latent_dim=latent_dim).to(device)
    print(pc_prior.input_layer_group.layer_0.params)
    
    # 2. Load Weights
    vae_model.load_state_dict(torch.load('cifar10/checkpoints/cifar_vae_weights.pth', map_location=device, weights_only=True))
    pc_prior.load_state_dict(torch.load('cifar10/checkpoints/cifar_pc_weights.pth', map_location=device, weights_only=True))
    
    vae_model.eval()
    
    num_samples = 64 # Generate an 8x8 grid

    with torch.no_grad():
        # 3. Exact Sampling via PyJuice
        # PyJuice natively handles the top-down sampling pass through the complex graph structure
        # returning exact samples from the learned multimodal joint distribution
        z_samples = pc_prior.sample(num_samples=num_samples).to(device)
        
        # 4. Decode the exact latents
        generated_imgs = vae_model.decode(z_samples).cpu()

    # 5. Plotting via torchvision's make_grid
    # generated_imgs shape is [64, 3, 32, 32]
    grid = make_grid(generated_imgs, nrow=8, padding=2, normalize=False)
    
    # Convert from PyTorch format [C, H, W] to Matplotlib format [H, W, C]
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title("CIFAR-10 Exact Prior Samples")
    plt.tight_layout()
    
    plt.savefig('cifar10_pcvae_samples.png', bbox_inches='tight', pad_inches=0.1)
    print("Generation complete. Saved to cifar10_pcvae_samples.png")

if __name__ == "__main__":
    generate_cifar10()
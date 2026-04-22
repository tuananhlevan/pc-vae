import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import PC_VAE

def sample_from_pc_prior(pc_prior, num_samples, device):
    """
    Samples exact vectors from the trained Probabilistic Circuit.
    """
    with torch.no_grad():
        # 1. Get the learned probabilities for each of the 10 clusters
        cluster_probs = F.softmax(pc_prior.mixture_logits, dim=0)
        
        # 2. Sample component indices based on those probabilities
        # This decides which "island" each sample comes from
        component_indices = torch.multinomial(cluster_probs, num_samples, replacement=True)
        
        # 3. Gather the exact means and variances for the chosen clusters
        selected_means = pc_prior.means[component_indices]
        selected_log_vars = pc_prior.log_vars[component_indices]
        
        # 4. Sample the continuous z vectors from those specific leaf Gaussians
        std = torch.exp(0.5 * selected_log_vars)
        eps = torch.randn_like(std)
        z_samples = selected_means + eps * std
        
        return z_samples

def generate_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and load weights
    model = PC_VAE(latent_dim=16, num_components=10).to(device)
    model.load_state_dict(torch.load('checkpoints/pc_vae_mnist.pth', weights_only=True))
    model.eval()

    num_samples = 16

    # Sample exactly from the learned PC graph
    z_samples = sample_from_pc_prior(model.pc_prior, num_samples, device)
    
    # Decode the samples back into images
    with torch.no_grad():
        generated_imgs = model.decoder(z_samples).cpu()
    
    # Reshape for plotting
    generated_imgs = generated_imgs.view(-1, 28, 28)

    # Plotting
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(generated_imgs[i].numpy(), cmap='gray')
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('assets/pc_vae_samples.png')
    print("Generation complete. Image saved as pc_vae_samples.png")

if __name__ == "__main__":
    generate_images()
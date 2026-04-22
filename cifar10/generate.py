import torch
import pyjuice as juice
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from model import ResNetAE, build_hclt_prior

def generate_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 256
    print(f"Initializing Generator on {device}...")

    # 1. Load Frozen VAE
    vae_model = ResNetAE(latent_dim=latent_dim).to(device)
    vae_model.load_state_dict(torch.load('checkpoints/clean_ae_stage1.pth', map_location=device, weights_only=True))
    vae_model.eval()

    # 2. Rebuild HCLT Structure
    # (We load the static latents strictly to satisfy PyJuice's structural initialization requirement)
    print("Rebuilding HCLT structure...")
    data = torch.load('checkpoints/cifar10_static_latents.pt', map_location='cpu')
    pc_prior = build_hclt_prior(data['latents'][:10000], latent_dim, device)
    
    # Load Weights
    pc_prior.load_state_dict(torch.load('checkpoints/hclt_pc_weights.pth', map_location=device, weights_only=True))

    # 3. Apply Temperature Sampling Hack (1D Interleaved Tensors)
    temperature = 0.8
    print(f"Applying Temperature Annealing (T={temperature})...")
    with torch.no_grad():
        for layer in pc_prior.input_layer_group:
            if hasattr(layer, 'params') and layer.params is not None:
                # Multiply only the Standard Deviations (Odd Indices) by the temperature
                layer.params[1::2] *= temperature

    # 4. Sample & Decode
    num_samples = 64
    print("Sampling from PC Prior...")
    with torch.no_grad():
        z_samples = juice.queries.sample(pc_prior, num_samples=num_samples).to(device)
        
        # Manually push through ResNet Decoder
        h = vae_model.decoder_input(z_samples)
        h = h.view(-1, 128, 4, 4)
        generated_imgs = vae_model.decoder(h).cpu()

    # 5. Plot Grid
    grid = make_grid(generated_imgs, nrow=8, padding=2, normalize=False)
    grid_np = grid.permute(1, 2, 0).numpy()

    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.title("HCLT PC-VAE Generations (CIFAR-10)")
    plt.tight_layout()
    
    plt.savefig('assets/cifar10_pc_vae_samples.png', bbox_inches='tight', pad_inches=0.1)
    print("Success! Open 'assets/cifar10_pc_vae_samples.png'")

if __name__ == "__main__":
    import os
    os.makedirs('assets', exist_ok=True)
    generate_images()
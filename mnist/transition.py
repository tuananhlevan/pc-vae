import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PC_VAE # Ensure this imports your saved model

def plot_transition(model_path='checkpoints/pc_vae_mnist.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    model = PC_VAE(latent_dim=16, num_components=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    num_steps = 15 # Number of transition frames between the two digits

    with torch.no_grad():
        # 2. Sample two different "islands" (components) to transition between
        # We manually pick component 0 and component 1 for distinct digits
        c1, c2 = 0, 1 
        
        mean1 = model.pc_prior.means[c1:c1+1]
        logvar1 = model.pc_prior.log_vars[c1:c1+1]
        std1 = torch.exp(0.5 * logvar1)
        z1 = mean1 + torch.randn_like(std1) * std1

        mean2 = model.pc_prior.means[c2:c2+1]
        logvar2 = model.pc_prior.log_vars[c2:c2+1]
        std2 = torch.exp(0.5 * logvar2)
        z2 = mean2 + torch.randn_like(std2) * std2

        # 3. Create a linear interpolation (Lerp) path between z1 and z2
        alphas = torch.linspace(0, 1, steps=num_steps).to(device)
        z_path = torch.cat([(1 - a) * z1 + a * z2 for a in alphas], dim=0)

        # 4. Decode the images along the path
        generated_imgs = model.decoder(z_path).cpu().view(num_steps, 28, 28).numpy()

        # 5. Calculate the EXACT log-probability of the path under the PC Prior
        log_probs = model.pc_prior.exact_log_prob(z_path).cpu().numpy()

    # --- Plotting ---
    fig = plt.figure(figsize=(15, 6))
    
    # Top Row: The Decoded Images
    for i in range(num_steps):
        ax = fig.add_subplot(2, num_steps, i + 1)
        ax.imshow(generated_imgs[i], cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Start (Island A)")
        elif i == num_steps - 1:
            ax.set_title("End (Island B)")

    # Bottom Row: The Exact Probability Graph
    ax_prob = fig.add_subplot(2, 1, 2)
    ax_prob.plot(np.linspace(0, 1, num_steps), log_probs, marker='o', color='b', linewidth=2)
    ax_prob.set_title("Exact Prior Log-Likelihood $\log p_{pc}(z)$ Along Transition Path")
    ax_prob.set_xlabel("Interpolation Step $\\alpha$")
    ax_prob.set_ylabel("Log Probability")
    ax_prob.grid(True)
    
    # Highlight the "Probability Valley"
    ax_prob.axvspan(0.3, 0.7, color='red', alpha=0.1, label='The "Dead Zone" between clusters')
    ax_prob.legend()

    plt.tight_layout()
    plt.savefig('assets/pc_vae_transition.png')
    print("Transition plotted and saved to pc_vae_transition.png")

if __name__ == "__main__":
    plot_transition()
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import PC_VAE # Ensure this imports your saved model

def plot_2d_manifold(model_path='checkpoints/pc_vae_mnist.pth', n_steps=20):
    """
    Plots a 2D grid manifold by bilinearly interpolating between 4 cluster means 
    in the 16-dimensional latent space.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model
    model = PC_VAE(latent_dim=16, num_components=10).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    with torch.no_grad():
        # 2. Pick 4 distinct clusters to act as the corners of our 2D grid
        # You can change these indices (0 to 9) to see different digit combinations
        c_top_left = 0
        c_top_right = 1
        c_bottom_left = 2
        c_bottom_right = 3
        
        # Extract their exact mean vectors (we ignore variance here for a clean path)
        z_tl = model.pc_prior.means[c_top_left]
        z_tr = model.pc_prior.means[c_top_right]
        z_bl = model.pc_prior.means[c_bottom_left]
        z_br = model.pc_prior.means[c_bottom_right]

        # 3. Create the 2D Bilinear Interpolation Grid
        grid_z = []
        for i in range(n_steps): # Y-axis (Row)
            alpha_y = i / (n_steps - 1)
            
            row_z = []
            for j in range(n_steps): # X-axis (Col)
                alpha_x = j / (n_steps - 1)
                
                # Interpolate top and bottom edges horizontally
                top = (1 - alpha_x) * z_tl + alpha_x * z_tr
                bottom = (1 - alpha_x) * z_bl + alpha_x * z_br
                
                # Interpolate vertically between the edges
                z = (1 - alpha_y) * top + alpha_y * bottom
                row_z.append(z)
                
            grid_z.append(torch.stack(row_z))
            
        # Shape: [n_steps * n_steps, latent_dim]
        grid_z = torch.stack(grid_z).view(-1, 16).to(device)

        # 4. Decode the entire grid
        generated_imgs = model.decoder(grid_z).cpu().view(n_steps, n_steps, 28, 28).numpy()

    # --- Plotting the Grid ---
    # Stitch the images together into one large canvas
    figure = np.zeros((28 * n_steps, 28 * n_steps))
    
    for i in range(n_steps):
        for j in range(n_steps):
            figure[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = generated_imgs[i, j]

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.title("16D PC-VAE 2D Latent Slice (Bilinear Interpolation)")
    
    plt.tight_layout()
    plt.savefig('assets/pc_vae_2d_manifold.png', bbox_inches='tight', pad_inches=1)
    print("2D Manifold plotted and saved to pc_vae_2d_manifold.png")

if __name__ == "__main__":
    plot_2d_manifold()
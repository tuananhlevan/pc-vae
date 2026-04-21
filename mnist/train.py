import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist.model import PC_VAE, pc_vae_loss # Assuming previous code is in model.py
import os

def train():
    # Setup device (Will automatically use your local CUDA environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3
    latent_dim = 16
    num_components = 10 # 10 mixture components for 10 digits

    # MNIST Dataset
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize Model and Optimizer
    model = PC_VAE(latent_dim=latent_dim, num_components=num_components).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            x_recon, mu, logvar, z = model(data)
            
            # Exact PC-VAE Loss
            loss = pc_vae_loss(x_recon, data, mu, logvar, z, model)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Negative ELBO: {avg_loss:.4f}")

    # Save the trained weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/pc_vae_mnist.pth')
    print("Training complete. Model saved to checkpoints/pc_vae_mnist.pth")

if __name__ == "__main__":
    train()
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Import our architecture from model.py
from cifar10.vae import ResNetVAE
from cifar10.pc_prior import build_pyjuice_prior
from cifar10.train import train_step

def train_cifar10():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Initializing training on: {device}")

    # Hyperparameters
    batch_size = 128
    epochs = 50
    learning_rate_vae = 1e-5
    learning_rate_pc = 1e-5
    latent_dim = 128

    # CIFAR-10 Dataset
    transform = transforms.ToTensor() # Keeps pixels in [0, 1] to match the Sigmoid output
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize Models
    vae_model = ResNetVAE(latent_dim=latent_dim).to(device)
    
    # Initialize and move PyJuice Circuit to GPU
    pc_prior = build_pyjuice_prior(latent_dim=latent_dim)
    pc_prior.to(device)

    # Separate Optimizers
    # PyJuice circuits register their parameters seamlessly with standard PyTorch optimizers
    optimizer_vae = optim.Adam(vae_model.parameters(), lr=learning_rate_vae)
    optimizer_pc = optim.Adam(pc_prior.parameters(), lr=learning_rate_pc)

    vae_model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            # Execute the combined exact training step
            loss = train_step(data, vae_model, pc_prior, optimizer_vae, optimizer_pc)
            total_loss += loss

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Negative ELBO: {avg_loss:.4f}")

    # Save Checkpoints
    os.makedirs('cifar10/checkpoints', exist_ok=True)
    torch.save(vae_model.state_dict(), 'cifar10/checkpoints/cifar_vae_weights.pth')
    
    # PyJuice has dedicated save/load methods to preserve graph structure
    torch.save(pc_prior.state_dict(), 'cifar10/checkpoints/cifar_pc_weights.pth')
    
    print("Training complete. Weights saved.")

if __name__ == "__main__":
    train_cifar10()
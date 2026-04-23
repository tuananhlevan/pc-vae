import torch
import torch.nn as nn
import torch.nn.functional as F
import pyjuice as juice
import pyjuice.nodes.distributions as dists

class EMAVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer('embedding', torch.empty(num_embeddings, embedding_dim))
        self.embedding.data.normal_()
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('embed_avg', self.embedding.data.clone())
        
        # --- THE FIX: ADD A STEP COUNTER ---
        self.register_buffer('step_counter', torch.tensor(0))

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        if self.training:
            # Advance the step counter
            self.step_counter += 1
            
            self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1 - self.decay)
            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

            # --- THE FIX: FREEZE RESTARTS AFTER WARMUP ---
            # Stop restarting dead codes after 2000 steps (roughly 5 epochs)
            # This forces the decoder to finally learn a stable codebook
            if self.step_counter.item() < 2000:
                dead_codes = self.cluster_size < 1.0
                if dead_codes.any():
                    num_dead = dead_codes.sum().item()
                    random_indices = torch.randint(0, flat_input.shape[0], (num_dead,), device=inputs.device)
                    self.embedding.data[dead_codes] = flat_input[random_indices].data
                    self.embed_avg.data[dead_codes] = flat_input[random_indices].data
                    self.cluster_size.data[dead_codes] = 1.0 

        quantized = F.embedding(encoding_indices.squeeze(1), self.embedding).view(input_shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        B = input_shape[0]
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, encoding_indices.view(B, -1)
    
# --- 1. NEW: The Residual Capacity Block ---
class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_residual_layers=2, num_residual_hiddens=32):
        super().__init__()
        self._layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(num_residual_hiddens, in_channels, kernel_size=1)
            ) for _ in range(num_residual_layers)
        ])

    def forward(self, x):
        for layer in self._layers:
            x = x + layer(x) # The true "ResNet" connection
        return F.leaky_relu(x, 0.2)


# --- 2. UPDATED: The Deep VQ-VAE ---
class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=128): # Bumped dim to 128
        super().__init__()
        
        # DEEP ENCODER: 32x32 -> 16x16 -> 8x8 + Residual Processing
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(in_channels=embedding_dim, num_residual_layers=2)
        )
        
        # Keep your exact EMA Quantizer here!
        self.vq = EMAVectorQuantizer(num_embeddings, embedding_dim)
        
        # DEEP DECODER: Residual Processing + 8x8 -> 16x16 -> 32x32
        self.decoder = nn.Sequential(
            ResidualStack(in_channels=embedding_dim, num_residual_layers=2),
            nn.Conv2d(embedding_dim, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        # 32x32 -> 16x16
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        # 16x16 -> 8x8
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        # # 8x8 -> 8x8
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, True)
        # )
        # # Output layer: 8x8 -> 8x8 single channel prediction map
        # self.layer4 = nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=1, padding=1)
        self.layer3 = nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # return self.layer4(x)
        return x

# ========= Latent Probabilistic Circuit =========
def build_discrete_hclt_prior(latent_data, num_cats=512, device='cuda'):
    """
    Constructs the HCLT Prior specifically for DISCRETE Categorical tokens.
    """
    ns = juice.structures.HCLT(
        latent_data.float().to(device), # PyJuice requires float for the MI tree builder
        
        # 1. MUST perfectly match the vocabulary size so no tokens are squashed together
        num_bins=num_cats,       
        
        # 2. Make sigma microscopic so the exact integer values aren't blurred
        sigma=0.01,              
        
        # 3. Exact number of variables in the 8x8 latent grid
        num_latents=64,   
        
        chunk_size=1,
        # THE DISCRETE SWITCH: Tell PyJuice to use exact Categorical probabilities
        input_dist=dists.Categorical(num_cats=num_cats)
    )
    
    ns.init_parameters(perturbation=2.0)
    pc_prior = juice.TensorCircuit(ns).to(device)
    return pc_prior
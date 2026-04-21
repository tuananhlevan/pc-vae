import torch
import pyjuice as pj
import pyjuice.nodes.distributions as dists

def build_pyjuice_prior(latent_dim=128, num_clusters=64):
    """
    Constructs a block-sparse Probabilistic Circuit using PyJuice.
    We create a deep mixture over the independent latent dimensions.
    """
    # 1. Define the Leaves (Continuous Normal Distributions)
    # We create a pool of 'num_clusters' (e.g., 64) different univariate 
    # Normal distributions for EACH latent dimension.
    leaves = [
        pj.inputs(v, num_node_blocks=num_clusters, dist=dists.Gaussian(mu=torch.randn(1)[0], sigma=torch.rand(1)[0] + 0.5, min_sigma=0.05))
        for v in range(latent_dim)
    ]
    
    # 2. The Product Layer (Independence)
    # We multiply the leaves together. In PyJuice, this creates a dense 
    # block of product nodes combining the distributions.
    # For a flat latent space, a single massive product layer is often enough.
    prod_nodes = pj.multiply(*leaves)
    
    # 3. The Root Sum Layer (Mixture Weights)
    # We sum everything into a single root node (num_node_groups=1).
    # This effectively creates a massive, highly flexible GMM, 
    # but strictly optimized via PyJuice's CUDA backend.
    root = pj.summate(prod_nodes, num_node_blocks=1)
    
    # 4. Compile into a TensorCircuit
    pc = pj.TensorCircuit(root)
    
    return pc
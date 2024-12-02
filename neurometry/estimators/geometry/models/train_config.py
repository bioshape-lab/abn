import torch

# hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# training
num_epochs = 200
batch_size = 32
lr = 1e-3

recon_weight = 1.0  # weight for the reconstruction loss
kld_weight = 0.03  # 0.03  # weight for KL loss
latent_weight = 100  # weight for latent regularization loss

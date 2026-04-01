import torch
import torch.nn.functional as F

def recon_loss_mse(recon, x):
    return F.mse_loss(recon, x, reduction="sum")

def weighted_mse_loss(recon, target, weight=10.0):
    # Create a mask: 1.0 where there is signal, 0.0 where background
    mask = (target > 0.01).float() 
    
    # Calculate squared error
    loss = (recon - target) ** 2
    
    # Multiply error by 'weight' where mask is 1, else keep as is
    weighted_loss = loss * (1 + (weight - 1) * mask)
    
    return weighted_loss.view(recon.size(0), -1).sum(dim=1).mean()

def kl_divergence(mu, logvar):
    # returns mean KL per batch (scalar)
    kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)
    return kl.mean()

def vae_loss(recon, x, mu, logvar, beta=1.0):
    r = weighted_mse_loss(recon, x)
    kl = kl_divergence(mu, logvar)
    return r + beta * kl, r, kl


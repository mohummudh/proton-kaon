import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from pathlib import Path

from src.models.vae import VAE
from src.losses.vae import vae_loss
from sklearn.model_selection import train_test_split

out = "/Volumes/easystore/proton-kaon/images/pk_48x48_log1p.pt"

data = torch.load(out, map_location="cpu")
p = data["p"]
k = data["k"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

epochs = 200
batch_size = 32
beta = 10.0
latent = 4

all_indices = np.arange(len(p))
train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# Initialize model, optimizer, loss
model = VAE(input_hw=(48, 48), latent=latent).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_losses = []
val_losses = []
train_recon, train_kl = [], []
val_recon, val_kl = [], []

print("===== Training (80/20 split) =====")
for epoch in range(epochs):
    model.train()
    total_train, tot_r, tot_kl = 0.0, 0.0, 0.0

    for xb in train_loader:
        xb = xb.to(device)
        optim.zero_grad()

        recon, mu, logvar, _ = model(xb)
        loss, r, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
        
        loss.backward()
        optim.step()
        
        total_train += loss.item()
        tot_r += r.item()
        tot_kl += kl.item()

    train_losses.append(total_train / len(train_loader))
    train_recon.append(tot_r / len(train_loader))
    train_kl.append(tot_kl / len(train_loader))

    model.eval()
    total_val, tot_r, tot_kl = 0.0, 0.0, 0.0
    with torch.no_grad():
        for xb in val_loader:
            xb = xb.to(device)
            recon, mu, logvar, _ = model(xb)
            loss, r, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
            total_val += loss.item()
            tot_r += r.item()
            tot_kl += kl.item()

    val_losses.append(total_val / len(val_loader))
    val_recon.append(tot_r / len(val_loader))
    val_kl.append(tot_kl / len(val_loader))

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train loss={train_losses[-1]:.6f} (recon={train_recon[-1]:.6f}, kl={train_kl[-1]:.6f}) | "
            f"Val loss={val_losses[-1]:.6f} (recon={val_recon[-1]:.6f}, kl={val_kl[-1]:.6f})"
        )


save_dir = Path("/Volumes/easystore/proton-kaon/models")
save_dir.mkdir(parents=True, exist_ok=True)
name = f"model_vae_latent{latent}_beta{beta}_epoch{epoch}.pt"
save_path = save_dir / name

# Save model
torch.save(model.state_dict(), save_path)
print(f"\nModel saved as {save_path}")
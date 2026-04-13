import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from pathlib import Path

from src.models.vae import VAE
from src.losses.vae import vae_loss
from src.train.train import train

from sklearn.model_selection import train_test_split

out = "/Volumes/easystore/proton-kaon/images/pk_48x48_log1p.pt"
data = torch.load(out, map_location="cpu")

p = data["p"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

EPOCHS = 200
BATCH_SIZE = 32
BETA = 10.0
LATENT = 4

all_indices = np.arange(len(p))
train_idx, val_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

model = VAE(input_hw=(48, 48), latent=LATENT).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

model, _, _, _, _, _, _ = train(device, train_loader, val_loader, model, optim, vae_loss, epochs=EPOCHS, beta=BETA)

save_dir = Path("/Volumes/easystore/proton-kaon/models")
save_dir.mkdir(parents=True, exist_ok=True)
name = f"model_vae_latent{LATENT}_beta{BETA}_epoch{EPOCHS}.pt"
save_path = save_dir / name

# Save model
torch.save(model.state_dict(), save_path)
print(f"\nModel saved as {save_path}")
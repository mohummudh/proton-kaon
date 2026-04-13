import torch

def train(device, train_loader, val_loader, model, optim, vae_loss, epochs, beta):
    train_losses = []
    val_losses = []
    train_recon, train_kl = [], []
    val_recon, val_kl = [], []

    print("===== Training =====")
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
    
    return model, train_losses, train_recon, train_kl, val_losses, val_recon, val_kl
import torch
import numpy as np

def inference(model, data):

    model.eval()

    device_inf = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Inference device:", device_inf)

    model.to(device_inf)

    X = torch.stack([data[i] for i in range(len(data))]).cpu().numpy()        # (N, 2, H, W)
    N, C, H, W = X.shape

    batch_size = 8   
    recon_all = np.empty((N, C, H, W), dtype=np.float32)

    latent_dim = model.fc_mu.out_features
    latent_vectors = np.empty((N, latent_dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            xb_np = X[i:j]                                 # (b, 2, H, W)
            xb = torch.tensor(xb_np, dtype=torch.float32).to(device_inf)

            recon, mu, logvar, z = model(xb)
            recon_all[i:j] = recon.cpu().numpy()
            latent_vectors[i:j] = mu.cpu().numpy()

            if device_inf.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    # per-sample MSE (RE) 
    mse_per_pixel = (recon_all - X) ** 2
    RE_per_sample = mse_per_pixel.mean(axis=(1, 2, 3))        # (N,)

    return latent_vectors, recon_all, RE_per_sample
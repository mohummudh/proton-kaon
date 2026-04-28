import json
from datetime import datetime

def save_run_log(cfg, device, train_subset, val_subset,
                 train_losses, train_recon, train_kl,
                 val_losses, val_recon, val_kl,
                 save_path):

    history = [
        {
            "epoch": i + 1,
            "train_loss": train_losses[i], "train_recon": train_recon[i], "train_kl": train_kl[i],
            "val_loss":   val_losses[i],   "val_recon":   val_recon[i],   "val_kl":   val_kl[i],
        }
        for i in range(len(train_losses))
    ]

    log = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "config": cfg,
        "dataset": {
            "path":     cfg["data"]["path"],
            "proton": cfg["data"]["proton"],
            "n_train":  len(train_subset),
            "n_val":    len(val_subset),
        },
        "history": history,
    }

    with open(save_path, "w") as f:
        json.dump(log, f, indent=2)

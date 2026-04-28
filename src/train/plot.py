import matplotlib.pyplot as plt

def plot_training(train_losses, train_recon, train_kl,
                  val_losses,   val_recon,   val_kl,
                  save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, tr, vl, title in zip(
        axes,
        [train_losses, train_recon, train_kl],
        [val_losses,   val_recon,   val_kl],
        ["Total loss", "Recon loss", "KL loss"]
    ):
        ax.plot(tr, label="train")
        ax.plot(vl, label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig

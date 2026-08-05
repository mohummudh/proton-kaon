from src.models.configVAE import VAE


def build_vae(cfg, device=None):
    """Construct the VAE described by a config.

    Single construction site for training, inference, and every analysis
    script, so no call site can drift from the architecture a checkpoint
    was trained with.
    """
    model = VAE(
        input_hw=tuple(cfg["model"]["input_hw"]),
        latent=cfg["model"]["latent"],
        channels=cfg["model"]["channels"],
        kernel=cfg["model"]["kernel"],
        stride=cfg["model"]["stride"],
        padding=cfg["model"]["padding"],
        activation=cfg["model"]["activation"],
        p_enc=cfg["model"].get("dropout", 0.0),
    )

    return model.to(device) if device is not None else model

def data_tag(cfg):
    """Optional `data.tag` marking a non-default train/val split of the same
    images (e.g. a balanced subsample). Keeps such runs from colliding with the
    default-split model of identical architecture."""
    tag = cfg["data"].get("tag")
    return f"_{tag}" if tag else ""


def seed_tag(cfg):
    """Optional `train.seed`, marking a run whose initialisation was seeded.

    Absent means unseeded — the historical behaviour, in which weight init,
    dropout and shuffle order come from process entropy. Omitting the tag in that
    case is deliberate: every model trained before seeding existed keeps exactly
    the name it already has, so its checkpoint, inference directory and figure
    directory all still resolve.
    """
    seed = (cfg.get("train") or {}).get("seed")
    return f"_seed{seed}" if seed is not None else ""


def split_filename(cfg):
    """Split file expected in output.splits_dir. Tagged data variants get their
    own split so they never silently reuse the default one.

    Deliberately free of seed_tag: a split is a property of the data, not of the
    training run, so N seeds over one split share one file rather than needing N
    identical copies.
    """
    return f"split_{cfg['data']['proton']}{data_tag(cfg)}.npz"


def model_name(cfg):
    """Model identity without the .pt extension — also the key for the
    per-model inference and figure directories."""
    species_tag = "_speciesall" if cfg["data"].get("proton") == "all" else ""

    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}{species_tag}{data_tag(cfg)}"
        f"{seed_tag(cfg)}"
    )


def model_filename(cfg):
    """Checkpoint filename: model_name plus the .pt extension."""
    return model_name(cfg) + ".pt"

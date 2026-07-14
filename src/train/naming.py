def model_filename(cfg):
    species_tag = "_speciesall" if cfg["data"].get("proton") == "all" else ""

    attn_cfg = cfg["model"].get("attention", {})
    attn_tag = ""
    if attn_cfg.get("enabled", False):
        stage = attn_cfg.get("after_stage")
        stage_str = "auto" if stage is None else str(stage)
        attn_tag = f"_attnS{stage_str}H{attn_cfg.get('heads', 4)}D{attn_cfg.get('depth', 2)}"

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
        f"_tx{cfg['data'].get('transform', 'none')}{species_tag}{attn_tag}.pt"
    )

import torch

# Most-probable ADC value (mode of max-ADC histogram) per plane.
# Collection peak ~500, induction peak ~250 — read from data histograms.

_COL_MPV = 483.0
_IND_MPV = 247.0

VALID_TRANSFORMS = {
    "none",
    "log1p",
    "sqrt",
    "cbrt",
    "minmax",
    "log1p_minmax",
    "clamp99_minmax",
    "tanh100",
    "tanh500",
    "mpv_linear",
    "mpv_tanh",
    "hill_mpv"
}

def apply_transform(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a preprocessing transform to a raw ADC tensor.

    x    : (..., H, W) non-negative float tensor of raw ADC values
    name : one of VALID_TRANSFORMS
    """
    if name == "none":
        return x

    elif name == "log1p":
        return torch.log1p(x)

    elif name == "sqrt":
        return x.clamp(min=0).sqrt()

    elif name == "cbrt":
        # cube root — gentler compression than sqrt, x is non-negative
        return x.clamp(min=0).pow(1 / 3)

    elif name == "minmax":
        # scale by dataset-level max so all values land in [0, 1]
        vmax = x.max()
        return x / (vmax + 1e-8)

    elif name == "log1p_minmax":
        t = torch.log1p(x)
        vmax = t.max()
        return t / (vmax + 1e-8)

    elif name == "clamp99_minmax":
        # clip at 99th percentile then scale to [0, 1] — suppresses hot-pixel extremes
        q99 = torch.quantile(x.float().reshape(-1), 0.99).item()
        return x.clamp(0, q99) / (q99 + 1e-8)

    elif name == "tanh100":
        # tanh scaled so that ADC ≈ 100 → 0.76; saturates above ~300–400
        return torch.tanh(x / 100.0)

    elif name == "tanh500":
        # softer saturation; ADC ≈ 500 → 0.76; suited for wider dynamic range
        return torch.tanh(x / 500.0)
    
    elif name == "mpv_linear":
        # Divide each channel by its most-probable ADC value so the mode maps to 1.0.
        # Fully linear — no compression. Bragg-peak outliers reach ~4–5.
        # Both channels end up on comparable scales despite the natural 2× difference.
        col = x[:, 0:1, :, :] / _COL_MPV
        ind = x[:, 1:2, :, :] / _IND_MPV
        return torch.cat([col, ind], dim=1)

    elif name == "mpv_tanh":
        # tanh with each channel's MPV as the saturation scale.
        # mode → tanh(1) ≈ 0.76; Bragg outliers smoothly approach 1.
        # Both channels land on [0, 1] with the same relative meaning.
        col = torch.tanh(x[:, 0:1, :, :] / _COL_MPV)
        ind = torch.tanh(x[:, 1:2, :, :] / _IND_MPV)
        return torch.cat([col, ind], dim=1)

    elif name == "hill_mpv":
        # Hill / Michaelis-Menten function: f(x) = x / (x + c).
        # f(MPV) = 0.5; approaches 1 asymptotically. Linear at low ADC (f≈x/c),
        # softer saturation than tanh. Each channel uses its own half-saturation.
        col = x[:, 0:1, :, :] / (x[:, 0:1, :, :] + _COL_MPV)
        ind = x[:, 1:2, :, :] / (x[:, 1:2, :, :] + _IND_MPV)
        return torch.cat([col, ind], dim=1)

    else:
        valid = ", ".join(sorted(VALID_TRANSFORMS))
        raise ValueError(f"Unknown transform {name!r}. Valid options: {valid}")

import torch

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

    else:
        valid = ", ".join(sorted(VALID_TRANSFORMS))
        raise ValueError(f"Unknown transform {name!r}. Valid options: {valid}")

import torch


def pick_device():
    """Best available torch device: CUDA on the training boxes, MPS on macOS.

    Every script picks its device here so a model trained on one backend is
    never silently run on another.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

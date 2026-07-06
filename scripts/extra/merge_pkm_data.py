"""One-off script: merge pk and muon image tensors into a single pkm file.

Run once before all-species training:
    python scripts/extra/merge_pkm_data.py
"""

import torch
from pathlib import Path

PK_PATH   = Path("/Volumes/easystore/proton-kaon/images/pk_48x48_raw_10-179wires.pt")
MUON_PATH = Path("/Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt")
OUT_PATH  = Path("/Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt")

pk   = torch.load(PK_PATH,   map_location="cpu")
muon = torch.load(MUON_PATH, map_location="cpu")

combined = {"p": pk["p"], "k": pk["k"], "m": muon["m"]}

for key, tensor in combined.items():
    print(f"  {key}: {tensor.shape}")

torch.save(combined, OUT_PATH)
print(f"Saved to {OUT_PATH}")

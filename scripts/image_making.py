import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pathlib import Path

from src.images import cut_start, pad_image, pad_image_batch_gpu, downsample_image
from src.cuts import image_cuts

LOG_PATH = Path("logs/image_making.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

col = pd.read_pickle('/Volumes/easystore/proton-kaon/clusters/col.pkl')
ind = pd.read_pickle('/Volumes/easystore/proton-kaon/clusters/ind.pkl')

col, ind = image_cuts(col, ind)
logger.info("Loaded %d collection rows and %d induction rows after image cuts", len(col), len(ind))

# making the images for the VAE model (GPU-accelerated)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", device)

# Separate by particle type
p_c_list = col[col['particle_type'] == 'proton']['image_intensity'].tolist()
p_i_list = ind[ind['particle_type'] == 'proton']['image_intensity'].tolist()
k_c_list = col[col['particle_type'] == 'kaon']['image_intensity'].tolist()
k_i_list = ind[ind['particle_type'] == 'kaon']['image_intensity'].tolist()

logger.info(
    "Processing %d proton collection, %d proton induction, %d kaon collection, %d kaon induction images",
    len(p_c_list),
    len(p_i_list),
    len(k_c_list),
    len(k_i_list),
)

# GPU batch padding
p_c = np.array(pad_image_batch_gpu(p_c_list, device=device, batch_size=64, cut_rows=50))
p_i = np.array(pad_image_batch_gpu(p_i_list, device=device, batch_size=64, cut_rows=50))
k_c = np.array(pad_image_batch_gpu(k_c_list, device=device, batch_size=64, cut_rows=50))
k_i = np.array(pad_image_batch_gpu(k_i_list, device=device, batch_size=64, cut_rows=50))

logger.info("Padded shapes: p_c=%s, p_i=%s, k_c=%s, k_i=%s", p_c.shape, p_i.shape, k_c.shape, k_i.shape)

# Downsample on GPU using PyTorch
p_c_tensor = torch.from_numpy(p_c).float().to(device)
p_i_tensor = torch.from_numpy(p_i).float().to(device)
k_c_tensor = torch.from_numpy(k_c).float().to(device)
k_i_tensor = torch.from_numpy(k_i).float().to(device)

# Use F.interpolate for GPU-accelerated downsampling
p_c_d = F.interpolate(p_c_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
p_i_d = F.interpolate(p_i_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
k_c_d = F.interpolate(k_c_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
k_i_d = F.interpolate(k_i_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

logger.info(
    "Downsampled shapes: p_c_d=%s, p_i_d=%s, k_c_d=%s, k_i_d=%s",
    p_c_d.shape,
    p_i_d.shape,
    k_c_d.shape,
    k_i_d.shape,
)

protimages = np.stack([p_c, p_i], axis=1)
kaonimages = np.stack([k_c, k_i], axis=1) 

p = np.stack([p_c_d, p_i_d], axis=1)            # shape: (N, 2, H, W)
k = np.stack([k_c_d, k_i_d], axis=1)            # shape: (N, 2, H, W)

p = torch.from_numpy(p).float()
k = torch.from_numpy(k).float()

p = torch.log1p(p)
k = torch.log1p(k)

out = "/Volumes/easystore/proton-kaon/images/pk_48x48_log1p.pt"

torch.save(
    {
        "p": p.cpu(),
        "k": k.cpu(),
    },
    out,
)

logger.info("Saved processed images to %s", out)

# to load:
# data = torch.load(out, map_location="cpu")
# p = data["p"]
# k = data["k"]

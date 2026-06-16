"""
Interactive UMAP latent-space explorer.

Click any point in the scatter plot to display the corresponding
collection and induction plane images for that event.

Usage:
    bokeh serve scripts/umap_explorer.py --args --config configs/your_config.yaml
    bokeh serve scripts/umap_explorer.py --args --config configs/your_config.yaml --muon

Then open http://localhost:5006/umap_explorer in your browser.
"""

import sys
import argparse
import pickle
import yaml
import numpy as np
import torch
from pathlib import Path

from bokeh.plotting import figure, curdoc
from bokeh.models import (
    ColumnDataSource, TapTool, HoverTool,
    LinearColorMapper, ColorBar, Div,
)
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True, help="Path to model YAML config")
parser.add_argument("--muon", action="store_true", help="Include muon points")
args = parser.parse_args(sys.argv[1:])

with open(args.config) as f:
    cfg = yaml.safe_load(f)

def _model_name(cfg):
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
        f"_tx{cfg['data'].get('transform', 'none')}"
    )

model_name = _model_name(cfg)
inf_dir    = Path(cfg["output"]["inference_dir"]) / model_name

# ── Load latents and UMAP reducer ─────────────────────────────────────────────
print("Loading latents...")
train_latents = np.load(inf_dir / "train.npz")["latents"]
val_latents   = np.load(inf_dir / "val.npz")["latents"]
kaon_latents  = np.load(inf_dir / "kaon.npz")["latents"]

muon_latents = None
if args.muon and (inf_dir / "muon.npz").exists():
    muon_latents = np.load(inf_dir / "muon.npz")["latents"]
    print(f"  muon latents: {len(muon_latents)}")

with open(inf_dir / "reducer.pkl", "rb") as f:
    reducer = pickle.load(f)

print("Transforming to UMAP...")
train_umap = reducer.transform(train_latents)
val_umap   = reducer.transform(val_latents)
kaon_umap  = reducer.transform(kaon_latents)
muon_umap  = reducer.transform(muon_latents) if muon_latents is not None else None

# ── Load images ───────────────────────────────────────────────────────────────
print("Loading images...")
pk_path   = Path(cfg["data"]["path"])
# Use raw (not log1p) for display — swap suffix if config points to log1p file
raw_path  = Path(str(pk_path).replace("log1p", "raw").replace("_log1p", "_raw"))
if not raw_path.exists():
    raw_path = pk_path  # fall back to whatever the config points to
pk_data   = torch.load(raw_path, map_location="cpu", weights_only=False)
p_images  = pk_data["p"].numpy()   # (N_p, 2, 48, 48)
k_images  = pk_data["k"].numpy()   # (N_k, 2, 48, 48)

split     = np.load(Path(cfg["output"]["splits_dir"]) / "split_p.npz")
train_idx = split["train_idx"]
val_idx   = split["val_idx"]

m_images  = None
if muon_latents is not None:
    muon_img_path = Path("/Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt")
    if muon_img_path.exists():
        m_images = torch.load(muon_img_path, map_location="cpu", weights_only=False)["m"].numpy()

print(f"  p: {p_images.shape}  k: {k_images.shape}" +
      (f"  m: {m_images.shape}" if m_images is not None else ""))

# ── Build combined point table ────────────────────────────────────────────────
COLORS = {
    "Proton (Train)": "#4C78A8",
    "Proton (Val)":   "#FB0019",
    "Kaon":           "#F58518",
    "Muon":           "#76B7B2",
}

xs, ys, particles, datasets, local_idxs, colors = [], [], [], [], [], []

for i, (x, y) in enumerate(train_umap):
    xs.append(float(x)); ys.append(float(y))
    particles.append("Proton (Train)"); datasets.append("train")
    local_idxs.append(i); colors.append(COLORS["Proton (Train)"])

for i, (x, y) in enumerate(val_umap):
    xs.append(float(x)); ys.append(float(y))
    particles.append("Proton (Val)"); datasets.append("val")
    local_idxs.append(i); colors.append(COLORS["Proton (Val)"])

for i, (x, y) in enumerate(kaon_umap):
    xs.append(float(x)); ys.append(float(y))
    particles.append("Kaon"); datasets.append("kaon")
    local_idxs.append(i); colors.append(COLORS["Kaon"])

if muon_umap is not None:
    for i, (x, y) in enumerate(muon_umap):
        xs.append(float(x)); ys.append(float(y))
        particles.append("Muon"); datasets.append("muon")
        local_idxs.append(i); colors.append(COLORS["Muon"])

source = ColumnDataSource(data=dict(
    x=xs, y=ys,
    particle=particles,
    dataset=datasets,
    local_idx=local_idxs,
    color=colors,
))

# ── UMAP scatter ──────────────────────────────────────────────────────────────
p_scatter = figure(
    width=700, height=580,
    title="UMAP Latent Space — click a point to view its image",
    tools="pan,wheel_zoom,reset,tap",
    active_scroll="wheel_zoom",
)
p_scatter.circle(
    x="x", y="y", color="color", alpha=0.6, size=5,
    source=source, line_width=0,
    selection_color="white", selection_line_color="black",
    selection_line_width=1, selection_alpha=1.0,
    nonselection_alpha=0.2,
)
p_scatter.add_tools(HoverTool(tooltips=[
    ("Particle", "@particle"),
    ("Index",    "@local_idx"),
]))
p_scatter.xaxis.axis_label = "UMAP 1"
p_scatter.yaxis.axis_label = "UMAP 2"

# ── Image panels ──────────────────────────────────────────────────────────────
blank = np.zeros((48, 48), dtype=np.float32)

img_source = ColumnDataSource(data=dict(collection=[blank], induction=[blank]))

mapper_col = LinearColorMapper(palette=Viridis256, low=0, high=1)
mapper_ind = LinearColorMapper(palette=Viridis256, low=0, high=1)

def _image_panel(title, key, mapper):
    p = figure(
        width=320, height=300, title=title,
        x_range=(0, 48), y_range=(0, 48),
        toolbar_location=None,
    )
    p.image(image=key, x=0, y=0, dw=48, dh=48,
            color_mapper=mapper, source=img_source)
    p.add_layout(ColorBar(color_mapper=mapper, width=8, label_standoff=6), "right")
    p.xaxis.axis_label = "Tick (time)"
    p.yaxis.axis_label = "Wire"
    return p

p_col = _image_panel("Collection Plane", "collection", mapper_col)
p_ind = _image_panel("Induction Plane",  "induction",  mapper_ind)

info_div = Div(
    text="<b>Click a point to view its image.</b>",
    width=660, styles={"font-size": "13px", "padding": "6px 0"},
)

# ── Tap callback ──────────────────────────────────────────────────────────────
def _get_image(dataset, local_idx):
    if dataset == "train":
        img = p_images[train_idx[local_idx]]
    elif dataset == "val":
        img = p_images[val_idx[local_idx]]
    elif dataset == "kaon":
        img = k_images[local_idx]
    elif dataset == "muon" and m_images is not None:
        img = m_images[local_idx]
    else:
        return blank, blank
    return img[0], img[1]   # collection (48×48), induction (48×48)

def on_tap(attr, old, new):
    if not new:
        return
    i        = new[0]
    ds       = source.data["dataset"][i]
    li       = source.data["local_idx"][i]
    particle = source.data["particle"][i]

    col_img, ind_img = _get_image(ds, li)

    img_source.data = dict(collection=[col_img], induction=[ind_img])

    col_max = float(col_img.max())
    ind_max = float(ind_img.max())
    mapper_col.high = max(col_max, 1.0)
    mapper_ind.high = max(ind_max, 1.0)

    info_div.text = (
        f"<b>{particle}</b> &nbsp;|&nbsp; dataset: {ds} &nbsp;|&nbsp; "
        f"index: {li} &nbsp;|&nbsp; "
        f"col max ADC: {col_max:.0f} &nbsp;|&nbsp; "
        f"ind max ADC: {ind_max:.0f}"
    )

source.selected.on_change("indices", on_tap)

# ── Layout ────────────────────────────────────────────────────────────────────
layout = column(
    info_div,
    row(p_scatter, column(p_col, p_ind)),
)
curdoc().add_root(layout)
curdoc().title = "UMAP Explorer"

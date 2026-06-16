"""
Manual kaon cluster labeler.

Shows all 432 csda-kaon events one at a time.  For each event you see all
collection-plane clusters (top row) and all induction-plane clusters (bottom
row).  Click "Select" under any cluster to mark it as the kaon candidate for
that plane.  The two planes are selected independently.

Navigation:
  ← Prev / Next →  — move between events
  ⏭ Next unset     — jump to the next event where either plane has no selection
  ✗ Mark no cluster — flag this event as having no usable cluster (skips it on save)
  💾 Save           — write csv_kaon_col_clean.pkl, csv_kaon_ind_clean.pkl,
                      csv_kaon_48x48_raw_clean.pt  (only events with both planes set)

Usage:
    bokeh serve scripts/extra/csda_kaon_labeler.py --args --config configs/...yaml
    # open http://localhost:5006/csda_kaon_labeler
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.images import pad_image_batch_gpu

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, Div, Button
from bokeh.layouts import row, column
from bokeh.palettes import Viridis256

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args(sys.argv[1:])

with open(args.config) as f:
    cfg = yaml.safe_load(f)

CLUST_DIR  = Path("/Volumes/easystore/proton-kaon/clusters")
IMG_DIR    = Path("/Volumes/easystore/proton-kaon/images")
FEAT_DIR   = Path("/Volumes/easystore/proton-kaon/features")
CSV_PATH   = Path("/Volumes/easystore/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv")

PAGE_SIZE  = 6    # clusters shown per plane at once
THUMB_H    = 80   # display height (wire axis)
THUMB_W    = 200  # display width  (time axis)

# ── Load clusters ─────────────────────────────────────────────────────────────
print("Loading unmatched clusters …")
col_df = pd.read_pickle(CLUST_DIR / "csv_kaon_col_unmatched.pkl").reset_index(drop=True)
ind_df = pd.read_pickle(CLUST_DIR / "csv_kaon_ind_unmatched.pkl").reset_index(drop=True)
print(f"  collection: {len(col_df)} clusters")
print(f"  induction:  {len(ind_df)} clusters")

# ── Build event list from CSV (all 432 events, fixed order) ──────────────────
target_df  = pd.read_csv(CSV_PATH)
event_list = list(zip(target_df['run'], target_df['subrun'], target_df['event']))
event_list = list(dict.fromkeys(event_list))   # deduplicate, preserve order
n_events   = len(event_list)
print(f"  {n_events} target events from CSV")

# Index clusters by event
def _build_index(df):
    idx = {}
    for ev_key, grp in df.groupby(['run', 'subrun', 'event']):
        rows = grp.sort_values('height', ascending=False).index.tolist()
        idx[ev_key] = rows
    return idx

col_idx = _build_index(col_df)
ind_idx = _build_index(ind_df)

# ── Thumbnail helper ──────────────────────────────────────────────────────────
def _thumb(img_array):
    """Resize (H, W) array to (THUMB_H, THUMB_W) for display."""
    arr = np.array(img_array, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return np.zeros((THUMB_H, THUMB_W), dtype=np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=(THUMB_H, THUMB_W), mode='bilinear', align_corners=False)
    return t.squeeze().numpy()

# ── State ─────────────────────────────────────────────────────────────────────
state = {
    'ev_idx':   0,
    'col_page': 0,
    'ind_page': 0,
}

# selections[ev_key] = {'col': row_idx or None, 'ind': row_idx or None, 'skip': bool}
selections = {k: {'col': None, 'ind': None, 'skip': False} for k in event_list}

# Auto-initialise: tallest cluster per plane per event
for ev_key in event_list:
    if ev_key in col_idx and col_idx[ev_key]:
        selections[ev_key]['col'] = col_idx[ev_key][0]
    if ev_key in ind_idx and ind_idx[ev_key]:
        selections[ev_key]['ind'] = ind_idx[ev_key][0]

# ── Build Bokeh widgets ───────────────────────────────────────────────────────
BLANK = np.zeros((THUMB_H, THUMB_W), dtype=np.float32)

def _make_slot(plane_label, slot_i):
    src     = ColumnDataSource(data=dict(image=[BLANK]))
    mapper  = LinearColorMapper(palette=Viridis256, low=0, high=1)
    fig     = figure(width=THUMB_W + 40, height=THUMB_H + 60,
                     x_range=(0, THUMB_W), y_range=(0, THUMB_H),
                     toolbar_location=None,
                     title=f"{plane_label} {slot_i}")
    fig.image(image="image", x=0, y=0, dw=THUMB_W, dh=THUMB_H,
              color_mapper=mapper, source=src)
    fig.add_layout(ColorBar(color_mapper=mapper, width=6, label_standoff=4), "right")
    fig.xaxis.visible = False
    fig.yaxis.visible = False
    info = Div(text="", width=THUMB_W + 40, styles={"font-size": "11px", "text-align": "center"})
    btn  = Button(label="Select", width=THUMB_W + 40, button_type="default")
    return src, mapper, fig, info, btn

col_slots = [_make_slot("Col", i) for i in range(PAGE_SIZE)]
ind_slots = [_make_slot("Ind", i) for i in range(PAGE_SIZE)]

header_div   = Div(text="", width=1200, styles={"font-size": "14px", "font-weight": "bold", "margin-bottom": "6px"})
col_hdr      = Div(text="<b>COLLECTION PLANE</b>", width=200, styles={"color": "#2166ac", "font-size": "13px"})
ind_hdr      = Div(text="<b>INDUCTION PLANE</b>",  width=200, styles={"color": "#1a9641", "font-size": "13px"})
col_pg_info  = Div(text="", width=200, styles={"font-size": "11px"})
ind_pg_info  = Div(text="", width=200, styles={"font-size": "11px"})
status_div   = Div(text="", width=1200, styles={"font-size": "12px", "margin-top": "6px"})

prev_btn     = Button(label="← Prev",          width=110)
next_btn     = Button(label="Next →",           width=110)
skip_btn     = Button(label="⏭ Next unset",    width=140)
mark_btn     = Button(label="✗ No cluster",    width=130, button_type="warning")
save_btn     = Button(label="💾 Save",          width=120, button_type="success")
col_prev_btn = Button(label="‹ clusters",       width=110)
col_next_btn = Button(label="clusters ›",       width=110)
ind_prev_btn = Button(label="‹ clusters",       width=110)
ind_next_btn = Button(label="clusters ›",       width=110)

# ── Render ────────────────────────────────────────────────────────────────────
def _render_plane(plane, slots, page, ev_key, sel_row):
    """Fill PAGE_SIZE slots for one plane, return total cluster count."""
    rows = col_idx[ev_key] if plane == 'col' else ind_idx[ev_key]
    rows = rows if rows else []
    n_total = len(rows)
    start   = page * PAGE_SIZE

    for slot_i, (src, mapper, fig, info, btn) in enumerate(slots):
        abs_i = start + slot_i
        if abs_i < n_total:
            row_idx = rows[abs_i]
            df      = col_df if plane == 'col' else ind_df
            img_raw = df.at[row_idx, 'image_intensity']
            h       = df.at[row_idx, 'height']
            w       = df.at[row_idx, 'width']
            thumb   = _thumb(img_raw)
            src.data = dict(image=[thumb])
            mapper.high = max(float(thumb.max()), 1.0)
            fig.title.text = f"{'Col' if plane=='col' else 'Ind'} #{abs_i}  h={h:.0f} w={w:.0f}"
            fig.visible   = True
            info.text     = f"row {row_idx}"
            info.visible  = True

            is_sel = (row_idx == sel_row)
            btn.button_type = "success" if is_sel else "default"
            btn.label       = "✓ Selected" if is_sel else "Select"
            btn.visible     = True
        else:
            src.data = dict(image=[BLANK])
            fig.visible  = False
            info.visible = False
            btn.visible  = False

    return n_total

def update_display():
    ev_idx  = state['ev_idx']
    ev_key  = event_list[ev_idx]
    sel     = selections[ev_key]
    n_done  = sum(1 for s in selections.values()
                  if s['skip'] or (s['col'] is not None and s['ind'] is not None))

    skipped = sel['skip']
    col_sel = sel['col']
    ind_sel = sel['ind']

    col_status = "✓" if col_sel is not None else "—"
    ind_status = "✓" if ind_sel is not None else "—"
    skip_flag  = "  ⚠ MARKED NO CLUSTER" if skipped else ""
    run, sub, ev = ev_key
    header_div.text = (
        f"Event {ev_idx+1} / {n_events} &nbsp;|&nbsp; "
        f"run={run} subrun={sub} event={ev} &nbsp;|&nbsp; "
        f"Col: {col_status} &nbsp; Ind: {ind_status}"
        f"<span style='color:orange'>{skip_flag}</span>"
    )

    n_col = _render_plane('col', col_slots, state['col_page'], ev_key, col_sel)
    n_ind = _render_plane('ind', ind_slots, state['ind_page'], ev_key, ind_sel)

    col_pg_info.text = (
        f"page {state['col_page']+1}/{max(1,(n_col+PAGE_SIZE-1)//PAGE_SIZE)}"
        f" ({n_col} clusters)"
    )
    ind_pg_info.text = (
        f"page {state['ind_page']+1}/{max(1,(n_ind+PAGE_SIZE-1)//PAGE_SIZE)}"
        f" ({n_ind} clusters)"
    )

    n_both  = sum(1 for s in selections.values() if s['col'] is not None and s['ind'] is not None)
    n_skip  = sum(1 for s in selections.values() if s['skip'])
    status_div.text = (
        f"Progress: {n_done}/{n_events} set &nbsp;|&nbsp; "
        f"both planes: {n_both} &nbsp;|&nbsp; "
        f"skipped: {n_skip} &nbsp;|&nbsp; "
        f"remaining: {n_events - n_done}"
    )

update_display()

# ── Callbacks ─────────────────────────────────────────────────────────────────
def _make_col_select(slot_i):
    def cb():
        ev_key = event_list[state['ev_idx']]
        rows   = col_idx.get(ev_key, [])
        abs_i  = state['col_page'] * PAGE_SIZE + slot_i
        if abs_i < len(rows):
            selections[ev_key]['col'] = rows[abs_i]
        update_display()
    return cb

def _make_ind_select(slot_i):
    def cb():
        ev_key = event_list[state['ev_idx']]
        rows   = ind_idx.get(ev_key, [])
        abs_i  = state['ind_page'] * PAGE_SIZE + slot_i
        if abs_i < len(rows):
            selections[ev_key]['ind'] = rows[abs_i]
        update_display()
    return cb

for i, (_, _, _, _, btn) in enumerate(col_slots):
    btn.on_click(_make_col_select(i))
for i, (_, _, _, _, btn) in enumerate(ind_slots):
    btn.on_click(_make_ind_select(i))

def go_prev():
    state['ev_idx']   = max(0, state['ev_idx'] - 1)
    state['col_page'] = 0
    state['ind_page'] = 0
    update_display()

def go_next():
    state['ev_idx']   = min(n_events - 1, state['ev_idx'] + 1)
    state['col_page'] = 0
    state['ind_page'] = 0
    update_display()

def go_skip_unset():
    for i in range(state['ev_idx'] + 1, n_events):
        s = selections[event_list[i]]
        if not s['skip'] and (s['col'] is None or s['ind'] is None):
            state['ev_idx']   = i
            state['col_page'] = 0
            state['ind_page'] = 0
            update_display()
            return
    status_div.text = "✓ No more unset events!"

def mark_no_cluster():
    ev_key = event_list[state['ev_idx']]
    selections[ev_key]['skip'] = not selections[ev_key]['skip']  # toggle
    update_display()

def col_prev_page():
    ev_key = event_list[state['ev_idx']]
    n = len(col_idx.get(ev_key, []))
    state['col_page'] = max(0, state['col_page'] - 1)
    update_display()

def col_next_page():
    ev_key = event_list[state['ev_idx']]
    n = len(col_idx.get(ev_key, []))
    max_page = max(0, (n - 1) // PAGE_SIZE)
    state['col_page'] = min(max_page, state['col_page'] + 1)
    update_display()

def ind_prev_page():
    state['ind_page'] = max(0, state['ind_page'] - 1)
    update_display()

def ind_next_page():
    ev_key = event_list[state['ev_idx']]
    n = len(ind_idx.get(ev_key, []))
    max_page = max(0, (n - 1) // PAGE_SIZE)
    state['ind_page'] = min(max_page, state['ind_page'] + 1)
    update_display()

def do_save():
    col_rows, ind_rows = [], []
    for ev_key in event_list:
        s = selections[ev_key]
        if s['skip'] or s['col'] is None or s['ind'] is None:
            continue
        col_rows.append(s['col'])
        ind_rows.append(s['ind'])

    if not col_rows:
        status_div.text = "⚠ Nothing to save — select clusters first."
        return

    col_clean = col_df.iloc[col_rows].reset_index(drop=True)
    ind_clean = ind_df.iloc[ind_rows].reset_index(drop=True)
    col_clean.to_pickle(CLUST_DIR / "csv_kaon_col_clean.pkl")
    ind_clean.to_pickle(CLUST_DIR / "csv_kaon_ind_clean.pkl")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    c_pad  = np.array(pad_image_batch_gpu(col_clean['image_intensity'].tolist(),
                                          device=device, batch_size=64, cut_rows=50))
    i_pad  = np.array(pad_image_batch_gpu(ind_clean['image_intensity'].tolist(),
                                          device=device, batch_size=64, cut_rows=50))
    c_t = F.interpolate(torch.from_numpy(c_pad).float().unsqueeze(1),
                        (48,48), mode='bilinear', align_corners=False).squeeze(1).numpy()
    i_t = F.interpolate(torch.from_numpy(i_pad).float().unsqueeze(1),
                        (48,48), mode='bilinear', align_corners=False).squeeze(1).numpy()
    imgs = np.stack([c_t, i_t], axis=1).astype(np.float32)
    torch.save({"k": torch.from_numpy(imgs)}, IMG_DIR / "csv_kaon_48x48_raw_clean.pt")

    status_div.text = (
        f"✓ Saved {len(col_rows)} events → "
        f"csv_kaon_*_clean.pkl + csv_kaon_48x48_raw_clean.pt"
    )
    print(f"Saved {len(col_rows)} cleaned events.")

prev_btn.on_click(go_prev)
next_btn.on_click(go_next)
skip_btn.on_click(go_skip_unset)
mark_btn.on_click(mark_no_cluster)
save_btn.on_click(do_save)
col_prev_btn.on_click(col_prev_page)
col_next_btn.on_click(col_next_page)
ind_prev_btn.on_click(ind_prev_page)
ind_next_btn.on_click(ind_next_page)

# ── Layout ────────────────────────────────────────────────────────────────────
def _slot_col(slots):
    cols = []
    for src, mapper, fig, info, btn in slots:
        cols.append(column(fig, info, btn))
    return row(*cols)

layout = column(
    header_div,
    row(col_hdr, col_prev_btn, col_pg_info, col_next_btn),
    _slot_col(col_slots),
    row(ind_hdr, ind_prev_btn, ind_pg_info, ind_next_btn),
    _slot_col(ind_slots),
    row(prev_btn, next_btn, skip_btn, mark_btn, save_btn),
    status_div,
)

curdoc().add_root(layout)
curdoc().title = "csda-kaon Labeler"

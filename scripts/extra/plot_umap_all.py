import argparse
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.")
    exit(1)

# ── Global matplotlib settings for publication ────────────────────────────────
plt.rcParams.update({
    # Font
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    # "text.usetex":        True,           # drop this line if no LaTeX install
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.title_fontsize": 8,
    # Lines / ticks
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    # Output
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# Paul Tol palette — colourblind-safe and greyscale-distinguishable
COLORS = {
    "Proton (Train)": "#0077BB",
    "Proton (Val)":   "#CC0000",
    "Kaon":           "#EE7733",
    "MIPs":           "#AA3377",
    "CSDA-Kaon":      "#CC3311",
}

# Species display names. Internally the third species is still keyed "muon"
# (filenames, npz keys, config); everything a reader sees says "MIPs", because
# the sample is a minimum-ionising selection rather than identified muons.
DISPLAY = {"proton": "Proton", "kaon": "Kaon", "muon": "MIPs"}

# Grey pair for a reference distribution drawn behind coloured foreground curves.
# Same values as plot_proxy_hists.py and cluster_latents.py — grey should mean the
# same thing in every figure in the paper.
TRAIN_FILL = "#D9D9D9"
TRAIN_EDGE = "#BDBDBD"

# Figure widths matching typical journal column widths (inches)
SINGLE_COL = 3.375   # ~86 mm  — single column
DOUBLE_COL = 6.875   # ~175 mm — double / full width


from src.train.naming import model_name as build_model_name


def resolve_dims(args_dims, n_latent: int, out_dir: Path) -> list:
    """Pick the two latent dims for the z-scatter plots: CLI override, else the
    top-2 single-dim AUCs from the logistic probe cache, else 4/7 (paper model)."""
    if args_dims is not None:
        return args_dims
    cache = out_dir / "cache_logistic.pkl"
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                results = pickle.load(f)["results"]
            singles = [(int(l[1:]), v["AUC"]) for l, v in results.items()
                       if l.startswith("z") and l[1:].isdigit()]
            if len(singles) >= 2:
                top = sorted(singles, key=lambda t: t[1], reverse=True)[:2]
                dims = sorted([top[0][0], top[1][0]])
                print(f"Auto-selected dims from logistic probe: z{dims[0]}, z{dims[1]} "
                      f"(AUC {top[0][1]:.3f}, {top[1][1]:.3f})")
                return dims
        except Exception as e:
            print(f"Could not auto-select dims from {cache.name}: {e}")
    return [4, 7] if n_latent > 7 else [0, max(0, min(1, n_latent - 1))]


def style_legend(leg, marker_size: float = 30):
    """Ensure legend markers are fully opaque and consistently sized."""
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([marker_size])


def make_legend_kwargs(loc: str = "best") -> dict:
    return dict(
        frameon=True,
        framealpha=0.85,
        edgecolor="0.75",
        loc=loc,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.35,
    )


def save(fig, path: Path):
    """Save as both PDF (paper) and PNG (quick preview)."""
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"  Saved {pdf_path.name}  +  {png_path.name}")


# ── paper figures ─────────────────────────────────────────────────────────────

def fig_umap_species_panel(species_embeds: list, out_dir: Path,
                           backdrop: bool = True):
    """The combined figure: one large all-species UMAP, three small ones beside it.

    Replaces the pair of separate figures (all-species scatter + per-species
    train/val grid). Train/val membership is deliberately *not* shown here —
    that question is answered by the two-sample tests instead, which leaves this
    figure making exactly one claim: the latent space organises itself by
    particle type. Each small panel therefore pools train and val for its
    species.

    All four panels share one set of axis limits, so a point sits at the same
    place in the small panel as in the large one. The grey backdrop repeats the
    full cloud behind each species, which is what makes "this species occupies
    *that* region" readable rather than just "this species is a blob".
    """
    all_pts = np.vstack([emb for _, emb, _ in species_embeds])
    fig = plt.figure(figsize=(DOUBLE_COL, round(DOUBLE_COL / 1.55, 3)))
    gs = fig.add_gridspec(3, 2, width_ratios=[3.15, 1], wspace=0.06, hspace=0.10)

    ax_big = fig.add_subplot(gs[:, 0])
    for name, emb, colour in species_embeds:
        ax_big.scatter(emb[:, 0], emb[:, 1], c=colour, label=name,
                       s=3, alpha=0.5, linewidths=0, rasterized=True)
    ax_big.set_xlabel("UMAP 1")
    ax_big.set_ylabel("UMAP 2")
    style_legend(ax_big.legend(**make_legend_kwargs(loc="upper right")))
    sns.despine(ax=ax_big)

    xlim, ylim = ax_big.get_xlim(), ax_big.get_ylim()
    for row, (name, emb, colour) in enumerate(species_embeds):
        ax = fig.add_subplot(gs[row, 1])
        if backdrop:
            ax.scatter(all_pts[:, 0], all_pts[:, 1], c="0.88",
                       s=1.2, alpha=0.6, linewidths=0, rasterized=True, zorder=1)
        ax.scatter(emb[:, 0], emb[:, 1], c=colour,
                   s=1.2, alpha=0.55, linewidths=0, rasterized=True, zorder=2)
        # Panel identity goes inside the axes: three stacked titles would eat
        # more vertical space than the panels themselves. The boxed background
        # is not decoration — the label sits on top of the scatter, and these
        # clouds reach into every corner of the frame.
        ax.annotate(f"{name} ({len(emb)})", xy=(0.035, 0.955),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=7.5, color=colour, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              alpha=0.88, edgecolor="none"))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("0.8")
    save(fig, out_dir / "umap_species_panel")
    plt.close(fig)


def val_scale_factor(n_train: int, n_val: int, mode: str) -> tuple:
    """How much to multiply the validation histogram by, and how to say so.

    The validation pool here is not 10% of the data — a species-balanced split
    caps the training set at the size of the smallest species and dumps every
    remaining image into validation, so val is the *larger* pool. Plotted as raw
    counts the two histograms differ by their sample sizes before they differ by
    anything interesting, which is the opposite of what the figure is for.

    'match' rescales val to the training count, so the two curves overlay and
    any difference left is a difference in shape. On a genuine 90/10 split this
    factor is exactly 9, which is why it is the default: the same rule carries
    over unchanged to the split sweep without changing the figure's meaning.
    """
    if mode == "none":
        return 1.0, "unscaled"
    if mode == "ninefold":
        return 9.0, r"$\times 9$ (nominal 90/10)"
    if mode == "tenth":
        return n_train / (9.0 * n_val), r"val at 1/9 of train"
    return n_train / n_val, rf"$\times${n_train / n_val:.2f} to train count"


def fig_recon_error_all_species(species_res: list, out_dir: Path,
                                val_scale: str = "match", bins: int = 40,
                                log_y: bool = True, background: str = "pooled"):
    """Reconstruction error for all three species on one axes, train vs val.

    One panel rather than three, because the point being made is a comparison
    *between* species (kaons reconstruct worse — they are the species the model
    finds hardest) as well as within each one (val sits on top of train). Three
    panels with independent y-axes make the first comparison impossible to read.

    TWO BACKGROUND MODES, ANSWERING DIFFERENT QUESTIONS

      background="pooled"  One solid grey shape: all three training sets summed,
            unlabelled. Reads as "here is the training distribution the model saw,
            and here is how the species decompose it". Same idiom as
            plot_proxy_hists.py and cluster_latents.py.

      background="split"   One translucent coloured fill per species' training set,
            so each validation line has its *own* training shape behind it in the
            same colour. Reads as "does each species' validation reproduce its own
            training distribution" — the per-species agreement the pooled version
            cannot show. Costs legibility: three fills overlap, and on a log axis
            the broad kaon fill spans the whole panel, so the fills are kept faint
            and each carries a stronger same-colour edge to stay traceable.

    Both are written on every run; they are alternatives for the same figure slot,
    not a figure and a supplement.

    ENCODING: in both modes colour identifies the species, a filled area is
    training and a solid saturated line is validation, rescaled onto training.

    WHY THE PER-SPECIES RESCALING IS STILL THE RIGHT ONE
        Each species' validation histogram is multiplied by its own
        n_train/n_val. That looks like a per-species choice sitting oddly under a
        pooled grey, but it is exactly what makes the two layers commensurate:
        scaling species i by n_train_i/n_val_i makes its curve integrate to
        n_train_i, so the three coloured curves sum to sum(n_train_i), which is
        the integral of the grey. The decomposition closes — to within the events
        outside the plotted range, since both layers are clipped at the 99th
        percentile of the pooled data (measured: 9297 against 9340, and at most
        ~3% of the peak bin).

    WHAT THIS FIGURE NO LONGER SHOWS
        With the grey pooled, a coloured line sitting below it is not a train/val
        disagreement — it is that species contributing only part of the total in
        that bin. Only where one species dominates the pool (the kaon tail) does
        "line follows grey" read directly as agreement. Per-species train/val
        agreement is therefore carried by the two-sample tests
        (scripts/latent_two_sample.py), not by this figure, which is the division
        of labour the 12 Aug notes asked for when they dropped the old figure 10.

        A single global factor n_train_total/n_val_total would also close in
        total, but each individual curve would then integrate to that species'
        *validation* share rather than its training count — and the balanced split
        makes those differ (training is exact thirds, the validation remainder is
        40/28/32). Per-species scaling divides that composition difference out, so
        a gap between a coloured line and the grey means a shape difference rather
        than a mixture difference.
    """
    re_max = np.percentile(np.concatenate([np.concatenate([tr, va])
                                           for _, tr, va, _ in species_res]), 99)
    edges = np.linspace(0, re_max, bins + 1)

    counts = []
    for name, tr_re, va_re, colour in species_res:
        tr_c = np.histogram(tr_re, bins=edges)[0]
        va_c = np.histogram(va_re, bins=edges)[0]
        factor, _ = val_scale_factor(len(tr_re), len(va_re), val_scale)
        counts.append({"name": name, "colour": colour, "train": tr_c,
                       "val": va_c * factor, "n_train": len(tr_re),
                       "n_val": len(va_re), "factor": factor})

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62,
                                    round(DOUBLE_COL * 0.62 / 1.45, 3)))
    train_total = np.sum([c["train"] for c in counts], axis=0)
    if background == "pooled":
        # One pooled grey shape, no edge, no per-species breakdown.
        ax.stairs(train_total, edges, fill=True, color=TRAIN_FILL,
                  linewidth=0, zorder=1)
    else:
        # Widest distribution to the back so the narrow proton and MIP peaks are
        # not buried by the broad kaon fill; the same-colour edge is what keeps
        # each training shape traceable where the three fills overlap.
        for c in sorted(counts, key=lambda c: -c["train"].std()):
            ax.stairs(c["train"], edges, fill=True, color=c["colour"],
                      alpha=0.20, linewidth=0, zorder=1)
        for c in counts:
            ax.stairs(c["train"], edges, color=c["colour"], alpha=0.55,
                      linewidth=0.8, zorder=2)
    # Validation: solid, in the species colour, on top. In pooled mode these sum
    # to train_total by construction of the rescaling (see the docstring).
    for c in counts:
        ax.stairs(c["val"], edges, color=c["colour"], linewidth=1.1, zorder=4)

    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Counts" if val_scale == "none"
                  else "Counts (validation rescaled to train)")
    ax.set_xlim(edges[0], edges[-1])
    # Pooled mode's grey is the sum, so it sets the ceiling; split mode's tallest
    # single species is lower, and using the sum there would leave dead space.
    peak = (train_total.max() if background == "pooled"
            else max(c["train"].max() for c in counts))
    if log_y:
        ax.set_yscale("log")
        # headroom for the legend, in decades rather than linear units
        ax.set_ylim(0.7, peak * 8)
    else:
        ax.set_ylim(0, peak * 1.42)

    # Grey means train, colour means val, so the key needs one grey entry and one
    # coloured line per species — four rows rather than six, in one corner. The
    # rescale factor is not spelled out: it is n_train/n_val, and both are printed
    # on the row, so naming it as well only lengthens the legend.
    if background == "pooled":
        handles = [Patch(facecolor=TRAIN_FILL, edgecolor="none",
                         label=f"train, all species ({int(train_total.sum())})")]
        handles += [Line2D([], [], color=c["colour"], lw=1.1,
                           label=f"{c['name']} val  {c['n_train']} / {c['n_val']}")
                    for c in counts]
    else:
        # In split mode the fill is already the species colour, so a shared key
        # ("filled = train, line = val") plus one coloured row per species says it
        # without three extra swatches.
        handles = [Patch(facecolor="0.55", alpha=0.20, edgecolor="0.55",
                         linewidth=0.8, label="filled = train"),
                   Line2D([], [], color="0.35", lw=1.1, label="line = val")]
        handles += [Patch(facecolor=c["colour"], alpha=0.65, edgecolor="none",
                          label=f"{c['name']}  {c['n_train']} / {c['n_val']}")
                    for c in counts]
    leg = ax.legend(handles=handles, title="train / val events",
                    **make_legend_kwargs(loc="upper right"))
    leg.get_title().set_fontsize(7.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    # The pooled log version is the paper figure, so it takes the plain name.
    stem = ("recon_error_all_species"
            + ("" if background == "pooled" else "_split")
            + ("" if log_y else "_linear"))
    save(fig, out_dir / stem)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--no-csda", action="store_true",
                        help="Skip loading and plotting CSDA kaons")
    parser.add_argument("--double-col", action="store_true",
                        help="Use double-column figure width instead of single-column")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip UMAP transform; load pre-computed embeddings from cache_umap.npz")
    parser.add_argument("--dims", nargs=2, type=int, default=None, metavar=("ZA", "ZB"),
                        help="Latent dimensions for the direct z-scatter plots "
                             "(default: top-2 discriminating dims from the logistic probe cache, "
                             "falling back to 4 7)")
    parser.add_argument("--val-scale", choices=["match", "ninefold", "tenth", "none"],
                        default="match",
                        help="How the validation histogram is rescaled in the "
                             "reconstruction-error figure. 'match' multiplies val by "
                             "n_train/n_val so the two curves overlay (exactly 9 on a "
                             "true 90/10 split); 'ninefold' always uses 9; 'tenth' draws "
                             "val at 1/9 of the training count; 'none' plots raw counts.")
    parser.add_argument("--linear-y", action="store_true",
                        help="Linear y-axis on the reconstruction-error figure, written as "
                             "recon_error_all_species_linear. The default is log: the proton "
                             "and MIP peaks are ~4x the kaon plateau, so on a linear axis the "
                             "kaon shape is squashed flat.")
    parser.add_argument("--no-backdrop", action="store_true",
                        help="Omit the grey all-species backdrop behind the small "
                             "per-species panels of the combined UMAP figure")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = build_model_name(cfg)
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    out_dir = Path("figs") / model_name / "latents-features"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_w = DOUBLE_COL if args.double_col else SINGLE_COL
    # Height: golden-ratio-ish for scatter plots
    fig_h = round(fig_w / 1.25, 3)

    # ── Load latents ──────────────────────────────────────────────────────────
    try:
        train_latents = np.load(inf_dir / "train.npz")["latents"]
        val_latents   = np.load(inf_dir / "val.npz")["latents"]
        kaon_latents  = np.load(inf_dir / "kaon.npz")["latents"]
        print(
            f"Loaded {len(train_latents)} protons (train), "
            f"{len(val_latents)} protons (val), "
            f"{len(kaon_latents)} kaon candidates."
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find inference files in {inf_dir}\n{e}")
        return

    muon_latents = None
    muon_path = inf_dir / "muon.npz"
    if muon_path.exists():
        muon_latents = np.load(muon_path)["latents"]
        print(f"Loaded {len(muon_latents)} muons.")

    csda_kaon_latents = None
    csda_kaon_path = inf_dir / "csda_kaon.npz"
    if not args.no_csda and csda_kaon_path.exists():
        csda_kaon_latents = np.load(csda_kaon_path)["latents"]
        print(f"Loaded {len(csda_kaon_latents)} csda-kaons.")

    # ── UMAP reducer ─────────────────────────────────────────────────────────
    reducer_path = inf_dir / "reducer.pkl"
    if reducer_path.exists():
        with open(reducer_path, "rb") as f:
            reducer = pickle.load(f)
        print(f"Loaded existing UMAP reducer from {reducer_path}")
    else:
        print("Training new UMAP reducer (this may take a moment)...")
        all_latents_list = [train_latents, val_latents, kaon_latents]
        if muon_latents is not None:
            all_latents_list.append(muon_latents)
        if csda_kaon_latents is not None:
            all_latents_list.append(csda_kaon_latents)
        all_latents = np.vstack(all_latents_list)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        reducer.fit(all_latents)
        with open(reducer_path, "wb") as f:
            pickle.dump(reducer, f)
        print("Fitted and saved new UMAP reducer.")

    embed_cache = out_dir / "cache_umap.npz"
    if args.from_cache and embed_cache.exists():
        print(f"Loading UMAP embeddings from cache ({embed_cache.name})...")
        _emb = np.load(embed_cache)
        train_umap     = _emb["train"]
        val_umap       = _emb["val"]
        kaon_umap      = _emb["kaon"]
        muon_umap      = _emb["muon"]      if "muon"      in _emb else None
        csda_kaon_umap = _emb["csda_kaon"] if "csda_kaon" in _emb else None
    else:
        print("Transforming latents...")
        train_umap = reducer.transform(train_latents)
        val_umap   = reducer.transform(val_latents)
        kaon_umap  = reducer.transform(kaon_latents)
        muon_umap      = reducer.transform(muon_latents)      if muon_latents is not None      else None
        csda_kaon_umap = reducer.transform(csda_kaon_latents) if csda_kaon_latents is not None else None
        _save_dict = dict(train=train_umap, val=val_umap, kaon=kaon_umap)
        if muon_umap      is not None: _save_dict["muon"]      = muon_umap
        if csda_kaon_umap is not None: _save_dict["csda_kaon"] = csda_kaon_umap
        np.savez(embed_cache, **_save_dict)
        print(f"Saved UMAP embeddings → {embed_cache.name}")

    # Scatter kwargs shared across UMAP plots
    sc_main  = dict(s=3, alpha=0.5, linewidths=0)
    sc_csda  = dict(s=10, alpha=1.0, linewidths=0.3, edgecolors="white")

    print("Plotting...")

    # ── Plot 1: proton train vs val ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    ax1.scatter(train_umap[:, 0], train_umap[:, 1],
                c=COLORS["Proton (Train)"], label="Proton (train)", **sc_main)
    ax1.scatter(val_umap[:, 0], val_umap[:, 1],
                c=COLORS["Proton (Val)"], label="Proton (val)", **sc_main)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    style_legend(ax1.legend(**make_legend_kwargs()))
    sns.despine(ax=ax1)
    fig1.tight_layout()
    save(fig1, out_dir / "umap_proton_train_val")
    plt.close(fig1)

    # ── Plot 2: all species ───────────────────────────────────────────────────
    proton_umap = np.vstack([train_umap, val_umap])
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.scatter(proton_umap[:, 0], proton_umap[:, 1],
                c=COLORS["Proton (Train)"], label="Proton", **sc_main)
    ax2.scatter(kaon_umap[:, 0], kaon_umap[:, 1],
                c=COLORS["Kaon"], label="Kaon", **sc_main)
    if muon_umap is not None:
        ax2.scatter(muon_umap[:, 0], muon_umap[:, 1],
                    c=COLORS["MIPs"], label="MIPs", **sc_main)
    if csda_kaon_umap is not None:
        ax2.scatter(csda_kaon_umap[:, 0], csda_kaon_umap[:, 1],
                    c=COLORS["CSDA-Kaon"], label="CSDA-Kaon", **sc_csda)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    style_legend(ax2.legend(**make_legend_kwargs()))
    sns.despine(ax=ax2)
    fig2.tight_layout()
    save(fig2, out_dir / "umap_all_species")
    plt.close(fig2)

    # ── Plots 3 & 4: direct latent-dimension scatters ─────────────────────────
    n_latent = train_latents.shape[1]
    za, zb = resolve_dims(args.dims, n_latent, out_dir)
    if za >= n_latent or zb >= n_latent:
        print(f"Skipping z{za}/z{zb} scatter plots: model has only {n_latent} latent dims "
              f"(pick others with --dims).")
    else:
        # ── Plot 3: za vs zb — all species ────────────────────────────────────
        proton_latents = np.vstack([train_latents, val_latents])
        fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
        ax3.scatter(proton_latents[:, za], proton_latents[:, zb],
                    c=COLORS["Proton (Train)"], label="Proton", **sc_main)
        ax3.scatter(kaon_latents[:, za], kaon_latents[:, zb],
                    c=COLORS["Kaon"], label="Kaon", **sc_main)
        if muon_latents is not None:
            ax3.scatter(muon_latents[:, za], muon_latents[:, zb],
                        c=COLORS["MIPs"], label="MIPs", **sc_main)
        if csda_kaon_latents is not None:
            ax3.scatter(csda_kaon_latents[:, za], csda_kaon_latents[:, zb],
                        c=COLORS["CSDA-Kaon"], label="CSDA-Kaon", **sc_csda)
        ax3.set_xlabel(rf"$z_{{{za}}}$")
        ax3.set_ylabel(rf"$z_{{{zb}}}$")
        style_legend(ax3.legend(**make_legend_kwargs()))
        sns.despine(ax=ax3)
        fig3.tight_layout()
        save(fig3, out_dir / f"z{za}_vs_z{zb}_all_species")
        plt.close(fig3)

        # ── Plot 4: za vs zb — train vs val protons ──────────────────────────
        fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
        ax4.scatter(train_latents[:, za], train_latents[:, zb],
                    c=COLORS["Proton (Train)"], label="Proton (train)", **sc_main)
        ax4.scatter(val_latents[:, za], val_latents[:, zb],
                    c=COLORS["Proton (Val)"], label="Proton (val)", **sc_main)
        ax4.set_xlabel(rf"$z_{{{za}}}$")
        ax4.set_ylabel(rf"$z_{{{zb}}}$")
        style_legend(ax4.legend(**make_legend_kwargs()))
        sns.despine(ax=ax4)
        fig4.tight_layout()
        save(fig4, out_dir / f"z{za}_vs_z{zb}_proton_train_val")
        plt.close(fig4)

    # ── All-species extras: per-species train/val UMAP + recon errors ─────────
    ss_path = inf_dir / "species_split.npz"
    if ss_path.exists() and muon_umap is not None:
        ss = np.load(ss_path)
        species_umaps = [
            ("Proton", train_umap, val_umap, COLORS["Proton (Train)"]),
            ("Kaon",  kaon_umap[ss["k_train_idx"]], kaon_umap[ss["k_val_idx"]], COLORS["Kaon"]),
            ("MIPs",  muon_umap[ss["m_train_idx"]], muon_umap[ss["m_val_idx"]], COLORS["MIPs"]),
        ]

        # ── Paper figure: figs 8 and 9 merged into one ────────────────────────
        # Species embeddings pool train and val — the small panels answer "where
        # does this species live", not "did the split hold up".
        fig_umap_species_panel(
            [(name, np.vstack([tr, va]), colour)
             for name, tr, va, colour in species_umaps],
            out_dir, backdrop=not args.no_backdrop)

        # Plot 5 (appendix backup): does each species' val set live where its
        # train set does? Superseded in the paper by the two-sample tests, kept
        # so the separated version is on hand if a reviewer asks for it.
        fig5, axes5 = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6),
                                   sharex=True, sharey=True)
        for ax, (sp_name, tr_emb, va_emb, colour) in zip(axes5, species_umaps):
            ax.scatter(tr_emb[:, 0], tr_emb[:, 1], c="0.7",
                       label=f"train (n={len(tr_emb)})", **sc_main)
            ax.scatter(va_emb[:, 0], va_emb[:, 1], c=colour,
                       label=f"val (n={len(va_emb)})", **sc_main)
            ax.set_title(sp_name, fontsize=9)
            ax.set_xlabel("UMAP 1")
            style_legend(ax.legend(**make_legend_kwargs()), marker_size=15)
            sns.despine(ax=ax)
        axes5[0].set_ylabel("UMAP 2")
        fig5.tight_layout()
        save(fig5, out_dir / "umap_train_val_by_species")
        plt.close(fig5)

        # Plot 6 (appendix backup): the separated, density-normalised version of
        # the reconstruction-error figure, one panel per species.
        kaon_re = np.load(inf_dir / "kaon.npz")["re"]
        muon_re = np.load(inf_dir / "muon.npz")["re"]
        species_res = [
            ("Proton", np.load(inf_dir / "train.npz")["re"],
                       np.load(inf_dir / "val.npz")["re"],   COLORS["Proton (Train)"]),
            ("Kaon",  kaon_re[ss["k_train_idx"]], kaon_re[ss["k_val_idx"]], COLORS["Kaon"]),
            ("MIPs",  muon_re[ss["m_train_idx"]], muon_re[ss["m_val_idx"]], COLORS["MIPs"]),
        ]

        # ── Paper figure: all three species on one axes, counts, val rescaled ──
        for bg in ("pooled", "split"):
            fig_recon_error_all_species(species_res, out_dir, background=bg,
                                        val_scale=args.val_scale,
                                        log_y=not args.linear_y)

        re_max = np.percentile(np.concatenate([np.concatenate([tr, va])
                                               for _, tr, va, _ in species_res]), 99)
        bins = np.linspace(0, re_max, 41)

        fig6, axes6 = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4), sharey=True)
        for ax, (sp_name, tr_re, va_re, colour) in zip(axes6, species_res):
            ax.hist(tr_re, bins=bins, density=True, color="0.7", alpha=0.7,
                    histtype="stepfilled", label=f"train (n={len(tr_re)})")
            ax.hist(va_re, bins=bins, density=True, color=colour, alpha=0.55,
                    histtype="stepfilled", label=f"val (n={len(va_re)})")
            ax.hist(va_re, bins=bins, density=True, color=colour,
                    histtype="step", linewidth=1.2)
            ax.set_title(sp_name, fontsize=9)
            ax.set_xlabel("Reconstruction error")
            leg = ax.legend(**make_legend_kwargs())
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)
            sns.despine(ax=ax)
        axes6[0].set_ylabel("Density")
        fig6.tight_layout()
        save(fig6, out_dir / "recon_error_by_species")
        plt.close(fig6)

    print("Done.")


if __name__ == "__main__":
    main()

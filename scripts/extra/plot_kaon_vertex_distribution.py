"""
Extract all kaon clusters (no cuts) and plot vertex position distributions
to inform acceptance window bounds for cluster_cuts().

Vertex = (bbox_min_row, bbox_min_col + argmax(image_intensity[0]))
       = (wire of first active wire, time tick of ADC peak in that wire)
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.open_root import open_root
from src.clustering import extract_clusters

KAONS_ROOT = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"
TREE_NAME  = "ana/raw;352"
OUT_DIR    = Path("figs/vertex_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_vertex(row):
    return int(row["bbox_min_col"]) + int(np.argmax(row["image_intensity"][0]))


def plot_plane(df, plane_name, out_dir):
    wire = df["bbox_min_row"].values
    time = df.apply(compute_vertex, axis=1).values

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{plane_name} plane — vertex distribution (N={len(df)})")

    # scatter
    axes[0].scatter(time, wire, s=1, alpha=0.3, rasterized=True)
    axes[0].set_xlabel("Vertex time tick")
    axes[0].set_ylabel("Vertex wire")
    axes[0].set_title("Scatter")

    # 2D histogram
    h = axes[1].hist2d(time, wire, bins=[200, 50], cmap="viridis")
    fig.colorbar(h[3], ax=axes[1], label="count")
    axes[1].set_xlabel("Vertex time tick")
    axes[1].set_ylabel("Vertex wire")
    axes[1].set_title("Heatmap")

    plt.tight_layout()
    out = out_dir / f"vertex_{plane_name.lower()}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


def main():
    print("Opening ROOT file...")
    k_df = open_root(KAONS_ROOT, tree_name=TREE_NAME)
    print(f"Total events: {len(k_df)}")

    print("Extracting clusters (no cuts)...")
    clusters = extract_clusters(k_df, particle_type="kaon", threshold=15, tree_name=TREE_NAME)
    print(f"Total clusters: {len(clusters)}")

    col = clusters[clusters["plane"] == "collection"]
    ind = clusters[clusters["plane"] == "induction"]
    print(f"  Collection: {len(col)}  Induction: {len(ind)}")

    plot_plane(col, "Collection", OUT_DIR)
    plot_plane(ind, "Induction", OUT_DIR)


if __name__ == "__main__":
    main()

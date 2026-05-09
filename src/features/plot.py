import numpy as np
import matplotlib.pyplot as plt


def hist(feat_df, feature, bins=50, xlabel=None):
    protons = feat_df[feat_df['particle_type'] == 'proton'][feature].dropna()
    kaons   = feat_df[feat_df['particle_type'] == 'kaon'][feature].dropna()
    muons   = feat_df[feat_df['particle_type'] == 'muon'][feature].dropna()

    series = [s for s in [protons, kaons, muons] if len(s) > 0]
    combined_min = min(s.min() for s in series)
    combined_max = max(s.max() for s in series)
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(protons, bins=bin_edges, alpha=0.6, label='Proton', density=True)
    ax.hist(kaons,   bins=bin_edges, alpha=0.6, label='Kaon',   density=True)
    if len(muons) > 0:
        ax.hist(muons, bins=bin_edges, alpha=0.6, label='Muon', density=True)
    ax.set_xlabel(xlabel or feature)
    ax.set_ylabel("Density")
    ax.set_title(feature)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_umap(train_umap, train_features, val_umap, val_features, kaon_umap, kaon_features,
              feature_name, muon_umap=None, muon_features=None, figsize=None, cmap='viridis'):
    has_muons = muon_umap is not None and muon_features is not None and feature_name in muon_features.columns
    n_panels  = 4 if has_muons else 3
    figsize   = figsize or (n_panels * 8, 5)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize)

    datasets = [
        (train_umap, train_features, 'Training Protons'),
        (val_umap,   val_features,   'Val Protons'),
        (kaon_umap,  kaon_features,  'Kaon Candidates'),
    ]
    if has_muons:
        datasets.append((muon_umap, muon_features, 'Muons (≥180 wires)'))

    all_vals = np.concatenate([
        train_features[feature_name].dropna().values,
        val_features[feature_name].dropna().values,
        kaon_features[feature_name].dropna().values,
        *([ muon_features[feature_name].dropna().values] if has_muons else []),
    ])
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, (umap_emb, features_df, label) in zip(axes, datasets):
        feature_vals = features_df[feature_name].values
        scatter = ax.scatter(
            umap_emb[:, 0], umap_emb[:, 1],
            c=feature_vals, cmap=cmap,
            alpha=0.6, s=20, edgecolors='none',
            vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(label)

    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(feature_name)

    return fig, axes

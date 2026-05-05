import numpy as np
import matplotlib.pyplot as plt

def hist(col, feature, bins=50, xlabel=None):
    
    protons = col[col['particle_type'] == 'proton'][feature].dropna()
    kaons   = col[col['particle_type'] == 'kaon'][feature].dropna()

    combined_min = min(protons.min(), kaons.min())
    combined_max = max(protons.max(), kaons.max())
    bin_edges = np.linspace(combined_min, combined_max, bins + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(protons, bins=bin_edges, alpha=0.6, label='Proton', density=True)
    ax.hist(kaons,   bins=bin_edges, alpha=0.6, label='Kaon',   density=True)
    ax.set_xlabel(xlabel or feature)
    ax.set_ylabel("Density")
    ax.set_title(feature)
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_umap(train_umap, train_features, val_umap, val_features, kaon_umap, kaon_features,
              feature_name, figsize=(24, 5), cmap='viridis'):
    """
    Plot three UMAP embeddings (train, val, kaon) colored by a feature in a 1x3 subplot grid.

    Parameters:
    -----------
    train_umap, val_umap, kaon_umap : ndarray of shape (n_samples, 2)
        2D UMAP coordinates
    train_features, val_features, kaon_features : pd.DataFrame
        DataFrames with feature columns
    feature_name : str
        Column name in features_dfs to color by
    figsize : tuple, optional
        Figure size (default: (18, 5))
    cmap : str, optional
        Colormap name (default: 'viridis')

    Returns:
    --------
    fig, axes : matplotlib figure and axes array
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    datasets = [
        (train_umap, train_features, 'Training Protons'),
        (val_umap, val_features, 'Val Protons'),
        (kaon_umap, kaon_features, 'Kaon Candidates'),
    ]

    # Get global min/max for consistent color scaling
    all_vals = np.concatenate([
        train_features[feature_name].dropna().values,
        val_features[feature_name].dropna().values,
        kaon_features[feature_name].dropna().values,
    ])
    vmin, vmax = all_vals.min(), all_vals.max()

    for ax, (umap_emb, features_df, label) in zip(axes, datasets):
        feature_vals = features_df[feature_name].values

        scatter = ax.scatter(
            umap_emb[:, 0],
            umap_emb[:, 1],
            c=feature_vals,
            cmap=cmap,
            alpha=0.6,
            s=20,
            edgecolors='none',
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title(label)

    # Single colorbar for all three
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label(feature_name)

    return fig, axes

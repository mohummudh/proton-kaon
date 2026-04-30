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
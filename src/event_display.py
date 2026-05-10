import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

def is_valid_cluster(region, plane):
    """
    Check if a cluster would survive the standard pipeline cuts 
    (from cluster_cuts and image_cuts).
    Returns: (is_valid: bool, reason: str)
    """
    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc
    
    # 1. Size cuts (from cluster_cuts lower=10, upper=179, and image_cuts width<1500)
    if not (10 < height < 179):
        return False, f"Height ({height}) not in (10,179)"
    if width >= 1500:
        return False, f"Width ({width}) >= 1500"
        
    # 2. Column maxes cut
    column_maxes = region.image_intensity.max(axis=1)
    if len(set(column_maxes)) <= 1:
        return False, "Uniform col_maxes"
        
    # 3. Geometric / Beam window cuts
    if plane == 'collection':
        if not (12 < minr < 37):
            return False, f"min_row ({minr}) not in (12,37)"
        if not (789 < maxc < 1927):
            return False, f"max_col ({maxc}) not in (789,1927)"
    elif plane == 'induction':
        if not (11 < minr < 35):
            return False, f"min_row ({minr}) not in (11,35)"
        if not (786 < maxc < 1794):
            return False, f"max_col ({maxc}) not in (786,1794)"
            
    return True, ""

def plot_event_displays(event, run, subrun, event_num, save_path=None):
    """
    Generate a 2x2 grid showing raw and clustered ADC matrices for an event.
    Valid clusters are boxed in green, rejected clusters in red with rejection reason.
    
    Args:
        event (Event): The Event object containing collection and induction matrices.
        run (int): Run number.
        subrun (int): Subrun number.
        event_num (int): Event number.
        save_path (Path or str): Path to save the plot. If None, plt.show() is called.
    """
    c_matrix = event.collection
    i_matrix = event.induction
    
    # Run clustering with standard thresholds
    clabeled, cregions = event.connectedregions(c_matrix, threshold=15)
    ilabeled, iregions = event.connectedregions(i_matrix, threshold=7)
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    
    # Top row: Raw
    sns.heatmap(c_matrix.T, cmap="viridis", cbar_kws={'label': 'ADC Counts'}, ax=axs[0, 0])
    axs[0, 0].set_title("Collection Plane (Raw)")
    axs[0, 0].invert_yaxis()
    
    sns.heatmap(i_matrix.T, cmap="viridis", cbar_kws={'label': 'ADC Counts'}, ax=axs[0, 1])
    axs[0, 1].set_title("Induction Plane (Raw)")
    axs[0, 1].invert_yaxis()
    
    # Bottom row: Clusters
    sns.heatmap(c_matrix.T, cmap="viridis", cbar_kws={'label': 'ADC Counts'}, ax=axs[1, 0], alpha=0.5)
    axs[1, 0].set_title("Collection Plane (Clustered)\nGreen = Kept | Red = Rejected")
    axs[1, 0].invert_yaxis()
    
    if cregions:
        for idx, reg in enumerate(cregions):
            minr, minc, maxr, maxc = reg.bbox
            is_valid, reason = is_valid_cluster(reg, 'collection')
            color = 'lime' if is_valid else 'red'
            
            rect = patches.Rectangle((minr, minc), maxr - minr, maxc - minc,
                                     linewidth=2, edgecolor=color, facecolor='none')
            axs[1, 0].add_patch(rect)
            
            label_text = str(idx) if is_valid else f"{idx}: {reason}"
            # Add a white background to text for readability against the heatmap
            axs[1, 0].text(minr, minc - 5, label_text, color=color, fontsize=9, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    sns.heatmap(i_matrix.T, cmap="viridis", cbar_kws={'label': 'ADC Counts'}, ax=axs[1, 1], alpha=0.5)
    axs[1, 1].set_title("Induction Plane (Clustered)\nGreen = Kept | Red = Rejected")
    axs[1, 1].invert_yaxis()
    
    if iregions:
        for idx, reg in enumerate(iregions):
            minr, minc, maxr, maxc = reg.bbox
            is_valid, reason = is_valid_cluster(reg, 'induction')
            color = 'lime' if is_valid else 'red'
            
            rect = patches.Rectangle((minr, minc), maxr - minr, maxc - minc,
                                     linewidth=2, edgecolor=color, facecolor='none')
            axs[1, 1].add_patch(rect)
            
            label_text = str(idx) if is_valid else f"{idx}: {reason}"
            axs[1, 1].text(minr, minc - 5, label_text, color=color, fontsize=9, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    for ax in axs.flat:
        ax.set_xlabel("Wire Number (0-239)")
        ax.set_ylabel("Time Tick")
    
    fig.suptitle(f"Kaon Event: Run {run}, Subrun {subrun}, Event {event_num}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
    else:
        plt.show()

from src.chi2 import parse_array, filter_arrays

def cluster_cuts(clusters_df, lower=1, upper=179):

    clusters_df = clusters_df[(clusters_df['height'] > lower) & (clusters_df['height'] < upper)]
    clusters_df = clusters_df[clusters_df['column_maxes'].map(lambda x: len(set(x)) > 1)]

    collection = (
        (clusters_df['plane'] == 'collection') &
        (clusters_df['bbox_min_row'] > 12) & (clusters_df['bbox_min_row'] < 37) &
        (clusters_df['bbox_max_col'] > 789) & (clusters_df['bbox_max_col'] < 1927)
    )

    induction = (
        (clusters_df['plane'] == 'induction') &
        (clusters_df['bbox_min_row'] > 11) & (clusters_df['bbox_min_row'] < 35) &
        (clusters_df['bbox_max_col'] > 786) & (clusters_df['bbox_max_col'] < 1794)
    )

    clusters_df = clusters_df[collection | induction].reset_index(drop=True); print(clusters_df.shape)

    return clusters_df


def reco_track_cuts(trk):

    trk["trkrr"]   = trk["trkrr"].apply(parse_array)
    trk["trkdedx"] = trk["trkdedx"].apply(parse_array)
    trk = filter_arrays(trk)
    trk = trk[trk["trkrr"].apply(len) > 0]
    trk = trk[trk["trkrr"].apply(len) == trk["trkdedx"].apply(len)].reset_index(drop=True)
    
    return trk


def image_cuts(col, ind, lower=1, upper=179, width=1500):

    removed_indices_col = col[~((col['height'] > lower) & (col['height'] < upper) & (col['width'] < width))].index
    removed_indices_ind = ind[~((ind['height'] > lower) & (ind['height'] < upper) & (ind['width'] < width))].index

    removed_indices = removed_indices_col.union(removed_indices_ind)

    col_cut = col[~col.index.isin(removed_indices)]; print(col.shape)
    ind_cut = ind[~ind.index.isin(removed_indices)]; print(ind.shape)

    return col_cut, ind_cut
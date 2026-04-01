import pandas as pd

def _plane_masks(df):
    pl = df['plane'].str.lower()
    induction_mask = pl.str.contains('induction')
    collection_mask = pl.str.contains('collection')
    return induction_mask, collection_mask

def pair_clusters(df,
                  height_weight=1.0,
                  row_weight=1.0,
                  col_weight=1.0):
    """
    Pair clusters across planes per event by calculating scores for all possible pairs
    and greedily selecting the lowest-scoring (best) matches.
    
    For each event, creates all possible induction-collection cluster pairs,
    calculates a score based on spatial differences, and performs 1-to-1 matching
    by iteratively selecting the best (lowest score) available pair.
    
    Args:
        df: DataFrame with cluster data
        height_weight: weight for height difference in score
        row_weight: weight for row (wire) difference in score
        col_weight: weight for column (time) difference in score
    
    Returns:
        DataFrame with matched pairs and their scores
    """
    keys = ['run', 'subrun', 'event']
    
    # checks
    for k in keys:
        if k not in df.columns:
            raise ValueError(f"Missing required key column: {k}")

    req = ['bbox_min_row','bbox_max_row','bbox_min_col','bbox_max_col',
           'cluster_idx','plane','height']
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Get induction and collection plane data
    ind_mask, col_mask = _plane_masks(df)
    ind = df.loc[ind_mask, keys + ['cluster_idx','bbox_min_row','bbox_max_row',
                                   'bbox_min_col','bbox_max_col','height']].copy()
    col = df.loc[col_mask, keys + ['cluster_idx','bbox_min_row','bbox_max_row',
                                   'bbox_min_col','bbox_max_col','height']].copy()

    ind = ind.rename(columns={
        'cluster_idx':'cluster_idx_ind',
        'bbox_min_row':'ind_min_r','bbox_max_row':'ind_max_r',
        'bbox_min_col':'ind_min_c','bbox_max_col':'ind_max_c',
        'height':'ind_h'
    })
    col = col.rename(columns={
        'cluster_idx':'cluster_idx_col',
        'bbox_min_row':'col_min_r','bbox_max_row':'col_max_r',
        'bbox_min_col':'col_min_c','bbox_max_col':'col_max_c',
        'height':'col_h'
    })

    # Create all possible pairs for each event (cartesian product)
    pairs = ind.merge(col, on=keys, how='inner')

    if len(pairs) == 0:
        return pd.DataFrame(columns=keys + [
            'cluster_idx_ind','cluster_idx_col',
            'd_min_c','d_max_c','d_min_r','d_max_r','d_h','match_score'
        ])

    # Calculate deltas for all pairs
    pairs['d_min_c'] = pairs['col_min_c'] - pairs['ind_min_c']  # time diff
    pairs['d_max_c'] = pairs['col_max_c'] - pairs['ind_max_c']
    pairs['d_min_r'] = pairs['col_min_r'] - pairs['ind_min_r']  # wire diff
    pairs['d_max_r'] = pairs['col_max_r'] - pairs['ind_max_r']
    pairs['d_h']     = pairs['col_h']     - pairs['ind_h']      # height diff

    # Calculate match score based on spatial differences
    col_score = (pairs[['d_min_c','d_max_c']].pow(2).sum(axis=1)) * col_weight
    row_score = (pairs[['d_min_r','d_max_r']].pow(2).sum(axis=1)) * row_weight
    h_score   = (pairs['d_h'].pow(2)) * height_weight
    pairs['match_score'] = (col_score + row_score + h_score).astype(float)

    pairs = pairs.sort_values(keys + ['match_score',
                                      'cluster_idx_ind','cluster_idx_col']).reset_index(drop=True)

    # Greedy 1-to-1 matching per event: iteratively pick lowest score pair
    def _greedy_one_to_one(group):
        run, subrun, event = group.name
        g = group.sort_values('match_score').copy()

        used_ind, used_col, keep = set(), set(), []
        for _, row in g.iterrows():
            i, c = row['cluster_idx_ind'], row['cluster_idx_col']
            if i not in used_ind and c not in used_col:
                keep.append(True)
                used_ind.add(i)
                used_col.add(c)
            else:
                keep.append(False)

        g = g.loc[keep].copy()
        g['run'] = run
        g['subrun'] = subrun
        g['event'] = event
        return g

    matched = (
        pairs
        .groupby(keys, group_keys=False)
        .apply(_greedy_one_to_one)
        .reset_index(drop=True)
    )

    return matched[keys + [
        'cluster_idx_ind','cluster_idx_col',
        'd_min_c','d_max_c','d_min_r','d_max_r','d_h','match_score'
    ]]

def matching(clusters_df):
    pairs_1to1 = pair_clusters(clusters_df)

    ind_mask, col_mask = _plane_masks(clusters_df)
    ind_all = clusters_df.loc[ind_mask].copy()
    col_all = clusters_df.loc[col_mask].copy()

    induction_df = pairs_1to1.merge(
        ind_all,
        left_on=['run', 'subrun', 'event', 'cluster_idx_ind'],
        right_on=['run', 'subrun', 'event', 'cluster_idx'],
        how='left',
        suffixes=('', '_ind')
    )

    collection_df = pairs_1to1.merge(
        col_all,
        left_on=['run', 'subrun', 'event', 'cluster_idx_col'],
        right_on=['run', 'subrun', 'event', 'cluster_idx'],
        how='left',
        suffixes=('', '_col')
    )

    return collection_df, induction_df
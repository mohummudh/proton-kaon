def _plane_masks(df):
    pl = df['plane'].str.lower()
    induction_mask = pl.str.contains('induction')
    collection_mask = pl.str.contains('collection')
    return induction_mask, collection_mask

def pair_clusters(df,
                  col_delta_lo=0, col_delta_hi=125, # time difference
                  row_tol=20,                       # wire difference
                  height_tol=5,
                  one_to_one=False):
    """
    Pair clusters across planes per event where:
      - same (run, subrun, event)
      - collection bbox_min/max_col are 0–125 greater than induction’s
      - bbox rows within ±20
      - |height_collection - height_induction| ≤ height_tol (default 5)
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

    # masks: getting induction and collection plane data
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

    pairs = ind.merge(col, on=keys, how='inner')

    # Deltas
    pairs['d_min_c'] = pairs['col_min_c'] - pairs['ind_min_c'] # time diff
    pairs['d_max_c'] = pairs['col_max_c'] - pairs['ind_max_c']
    pairs['d_min_r'] = pairs['col_min_r'] - pairs['ind_min_r'] # wire diff
    pairs['d_max_r'] = pairs['col_max_r'] - pairs['ind_max_r']
    pairs['d_h']     = pairs['col_h']     - pairs['ind_h']     # height diff

    # rules
    cond = (
        (pairs['d_min_c'].between(col_delta_lo, col_delta_hi)) &
        (pairs['d_max_c'].between(col_delta_lo, col_delta_hi)) &
        (pairs['d_min_r'].abs() <= row_tol) &
        (pairs['d_max_r'].abs() <= row_tol) &
        pairs['ind_h'].notna() & pairs['col_h'].notna() &
        (pairs['d_h'].abs() <= height_tol)
    )

    candidates = pairs.loc[cond].copy()

    # Score: closeness to target column shift + row agreement + height agreement
    target_mid = (col_delta_lo + col_delta_hi) / 2.0
    col_score = (candidates[['d_min_c','d_max_c']] - target_mid).pow(2).sum(axis=1)
    row_score = candidates[['d_min_r','d_max_r']].pow(2).sum(axis=1)
    h_score   = candidates['d_h'].pow(2)
    candidates['match_score'] = (col_score + row_score + h_score).astype(float)

    candidates = candidates.sort_values(keys + ['match_score',
                                                'cluster_idx_ind','cluster_idx_col']).reset_index(drop=True)

    base_cols = keys + [
        'cluster_idx_ind','cluster_idx_col',
        'ind_min_r','ind_max_r','ind_min_c','ind_max_c','ind_h',
        'col_min_r','col_max_r','col_min_c','col_max_c','col_h',
        'd_min_c','d_max_c','d_min_r','d_max_r','d_h','match_score'
    ]

    if not one_to_one:
        return candidates[base_cols]

    # Greedy 1-1 per event
    def _greedy_one_to_one(group):
        g = group.sort_values('match_score').copy()
        used_ind, used_col, keep = set(), set(), []
        for _, row in g.iterrows():
            i, c = row['cluster_idx_ind'], row['cluster_idx_col']
            if i not in used_ind and c not in used_col:
                keep.append(True); used_ind.add(i); used_col.add(c)
            else:
                keep.append(False)
        return g.loc[keep]

    one2one = (
        candidates
        .groupby(keys, group_keys=False)
        .apply(_greedy_one_to_one)
        .reset_index(drop=True)
    )

    return one2one[keys + [
        'cluster_idx_ind','cluster_idx_col',
        'd_min_c','d_max_c','d_min_r','d_max_r','d_h','match_score'
    ]]
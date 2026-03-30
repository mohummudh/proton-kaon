import numpy as np
import pandas as pd

from tqdm.auto import tqdm

def parse_array(cell):
    if pd.isna(cell) or cell == "":
        return np.array([], dtype=np.float32)
    return np.array([float(x) for x in cell.split(";")], dtype=np.float32)

def filter_arrays(df):
    """Filter values > 100 from dedx arrays in the dataframe"""
    new_dedx = []
    new_rr = []
    
    for dedx, rr in zip(df['trkdedx'], df['trkrr']):
        indices = np.where(dedx > 100)[0]
        new_dedx.append(np.delete(dedx, indices))
        new_rr.append(np.delete(rr, indices))
    
    df['trkdedx'] = new_dedx
    df['trkrr'] = new_rr
    return df

def build_res_index(res: np.ndarray) -> dict:
    """
    Map exact residual-range values (rounded to 2dp) -> index in res.
    This emulates: np.where(res == ro)[0][0] but fast.
    """
    res = np.asarray(res)
    # Alek compares floats directly; to make it robust while matching intent,
    # we store keys at 2dp (same as ro).
    keys = np.round(res.astype(np.float64), 2)
    # If duplicates exist, first occurrence matches np.where(...)[0][0]
    idx = {}
    for i, k in enumerate(keys):
        if k not in idx:
            idx[float(k)] = i
    return idx

def chi2_track_alek(rr, dedx, res_to_idx, bb, step=0.05):
    """
    Match Alek:
      ro = round(round(rr/0.05)*0.05, 2)
      x_l = np.where(res == ro)[0][0]
      chi += ((dedx - bb[x_l])**2) / bb[x_l]
      return chi / len(rr)
    Differences vs your earlier code:
      - no nearest neighbour
      - denominator is bb[x_l] (no eps clamp)
      - divides by len(x) (original length), not 'used' count
      - does not filter invalid points (unless you want to exactly match him)
    """
    rr = np.asarray(rr)
    dedx = np.asarray(dedx)
    if rr.size == 0 or rr.size != dedx.size:
        return np.nan

    chi = 0.0
    n = len(rr)

    for r, d in zip(rr, dedx):
        ro = round(round(float(r) / step) * step, 2)
        # exact match lookup like np.where(res == ro)[0][0]
        x_l = res_to_idx.get(ro, None)
        if x_l is None:
            # Alek would crash here. To stay compatible, return NaN (or raise).
            return np.nan
        exp = float(bb[x_l])
        chi += ((float(d) - exp) ** 2) / exp

    return chi / n

def do_chi_squared(df: pd.DataFrame,
                                  res_k: np.ndarray, bb_k: np.ndarray,
                                  res_p: np.ndarray, bb_p: np.ndarray,
                                  step: float = 0.05) -> pd.DataFrame:
    df = df.copy()

    # Build exact-match maps once
    resk_map = build_res_index(res_k)
    resp_map = build_res_index(res_p)

    pairs = list(zip(df["trkrr"], df["trkdedx"]))

    df["chi_squared_kaon"] = [
        chi2_track_alek(rr, dx, resk_map, bb_k, step=step)
        for rr, dx in tqdm(pairs, total=len(pairs), desc="chi^2 kaon (Alek-match)")
    ]

    df["chi_squared_proton"] = [
        chi2_track_alek(rr, dx, resp_map, bb_p, step=step)
        for rr, dx in tqdm(pairs, total=len(pairs), desc="chi^2 proton (Alek-match)")
    ]

    # Alek: kaon if chi_k < chi_p else proton
    df["particle_hypothesis"] = np.where(
        df["chi_squared_kaon"] < df["chi_squared_proton"], 0, 1
    )

    return df



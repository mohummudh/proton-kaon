import numpy as np

from scipy.stats import entropy, skew, kurtosis
from scipy.signal import peak_widths, find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.measure import label, regionprops

def n_pixels(image):
    """Number of above-threshold pixels — proxy for track volume."""
    return np.sum(image > 0)

def profile_skewness(column_maxes):
    """Skewness of the dE/dx value distribution — right-skewed for protons with a Bragg spike pulling the tail high."""
    return skew(column_maxes[column_maxes > 0])

def profile_kurtosis(column_maxes):
    """Kurtosis of the dE/dx value distribution — higher for kaons with a flat, concentrated profile; lower for protons with a rising spread."""
    return np.log(kurtosis(column_maxes[column_maxes > 0]))

def n_local_maxima(column_maxes):
    """Number of local peaks in the dE/dx profile — kaons with secondary interactions or decay products may show multiple bumps."""
    cm = uniform_filter1d(column_maxes.astype(float), size=3)
    peaks, _ = find_peaks(cm)
    return len(peaks)

def solidity(image_intensity, threshold=0):
    """Area divided by convex hull area — deviations below 1.0 indicate kinks or bends, e.g. from kaon decay in flight."""
    binary = image_intensity > threshold
    labeled = label(binary)
    if labeled.max() == 0:
        return np.nan
    regions = regionprops(labeled)
    main = max(regions, key=lambda r: r.area)  # largest connected region
    return main.solidity
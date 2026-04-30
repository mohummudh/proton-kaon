import numpy as np

from scipy.stats import entropy, skew, kurtosis
from scipy.signal import peak_widths, find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.measure import label, regionprops

def n_pixels(image):
    return np.sum(image > 0)

def profile_skewness(column_maxes):
    return skew(column_maxes[column_maxes > 0])

def profile_kurtosis(column_maxes):
    return kurtosis(column_maxes[column_maxes > 0])

def n_local_maxima(column_maxes):
    peaks, _ = find_peaks(column_maxes)
    return len(peaks)

def solidity(image_intensity, threshold=0):
    binary = image_intensity > threshold
    labeled = label(binary)
    if labeled.max() == 0:
        return np.nan
    regions = regionprops(labeled)
    main = max(regions, key=lambda r: r.area)  # largest connected region
    return main.solidity
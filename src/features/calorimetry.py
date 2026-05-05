import numpy as np

from scipy.stats import entropy, skew, kurtosis
from scipy.signal import peak_widths, find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.measure import label, regionprops

def total_adc(image):
    """Log of total charge deposited — protons deposit more energy at the same momentum."""
    return np.log(np.sum(image))

def mean_adc(image):
    """Mean ADC across all pixels — average energy deposition per unit area."""
    return np.mean(image)

def median_adc(image):
    """Median ADC of signal pixels — robust estimate of typical dE/dx."""
    return np.median(image[image > 0])

def max_adc(image):
    """Peak ADC anywhere in the cluster — height of the Bragg spike."""
    return np.max(image)

def std_adc(image):
    """Standard deviation of signal pixel ADC values — spread of energy deposition."""
    return np.std(image[image > 0])

def adc_entropy(image, n_bins=50):
    """
    The number of bits needed to describe the spread of ADC values.
    High if energy is deposited at many different levels.
    Lower if most pixels share the same intensity.
    """

    pixels = image[image > 0].ravel()
    if len(pixels) < 2:
        return np.nan
    counts, _ = np.histogram(pixels, bins=n_bins)
    n_occupied = (counts > 0).sum()
    if n_occupied < 2:
        return 0.0
    return entropy(counts) / np.log(n_occupied)

def bragg_peak_height(column_maxes):
    """Maximum dE/dx across all wires — height of the Bragg peak."""
    return np.max(column_maxes)

def bragg_peak_position(column_maxes):
    """Normalised position of the peak dE/dx along the track — near 1.0 for protons stopping at the end."""
    return np.argmax(column_maxes) / len(column_maxes)

def bragg_peak_ratio(column_maxes):
    """Peak dE/dx divided by mean — prominence of the Bragg spike above the track average."""
    return np.max(column_maxes) / np.mean(column_maxes)

def bragg_peak_to_median(column_maxes):
    """Peak dE/dx divided by median — robust version of bragg_peak_ratio."""
    return np.max(column_maxes) / np.median(column_maxes)

def end_vs_start_ratio(column_maxes, p=0.1):
    """Mean dE/dx in the last 10% of wires over the first 10% — quantifies the Bragg rise."""
    n = len(column_maxes)
    k = int(p * n)
    return np.mean(column_maxes[-k:]) / np.mean(column_maxes[:k])

def last_quartile_mean(column_maxes, p=0.25):
    """Mean dE/dx in the final 25% of wires — elevated for protons approaching the Bragg peak."""
    n = len(column_maxes)
    k = int(p * n)
    return np.mean(column_maxes[-k:])

def first_quartile_mean(column_maxes, p=0.25):
    """Mean dE/dx in the first 25% of wires — baseline energy loss at track entry."""
    n = len(column_maxes)
    k = int(p * n)
    return np.mean(column_maxes[:k])

def bragg_rise_slope(column_maxes):
    """Linear fit slope of dE/dx along the track — steeper for protons rising to a Bragg peak."""
    x = np.arange((len(column_maxes)))
    slope, _ = np.polyfit(x, column_maxes, 1)
    return slope

def peak_integral_fraction(column_maxes, p=0.15):
    """
    The fraction of total track charge deposited in the last 15% of wires.
    High for protons with a Bragg spike at the end, and for stopping kaons,
    Low for kaons that decay.
    """

    total = np.sum(column_maxes[column_maxes > 0])
    if total == 0:
        return np.nan
    k = max(1, int(p * len(column_maxes)))
    end = np.sum(column_maxes[-k:])   # last k wires spatially
    return end / total

def bragg_peak_width(column_maxes):
    """FWHM of the peak in the dE/dx profile — narrow for a sharp proton Bragg spike, wide for a flat kaon profile."""
    peak_idx = np.argmax(column_maxes)
    widths, _, _, _ = peak_widths(column_maxes, [peak_idx], rel_height=0.5)
    return widths[0]

def profile_cv(column_maxes):
    """Coefficient of variation of dE/dx — high when energy loss is uneven along the track."""
    return np.std(column_maxes[column_maxes > 0]) / np.mean(column_maxes[column_maxes > 0])

def monotonic_rise_fraction(column_maxes, smooth=3, min_wires=10):
    """Fraction of consecutive wire pairs where dE/dx increases — near 1.0 for protons consistently building toward a Bragg peak."""
    if len(column_maxes) < min_wires:
        return np.nan
    cm = uniform_filter1d(column_maxes.astype(float), size=smooth)
    diffs = np.diff(cm)
    return (diffs > 0).sum() / len(diffs)

def relative_peak_energy(column_maxes):
    """Fraction of total charge within a window around the peak wire — concentrated for a sharp Bragg spike."""
    i = np.argmax(column_maxes)
    l = int(len(column_maxes) * 0.1)
    window = column_maxes[max(0, i-l): i+l+1]
    peak = np.sum(window)
    total = np.sum(column_maxes[column_maxes>0])
    return peak / total



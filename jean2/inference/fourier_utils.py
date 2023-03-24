"""
Created on 22/03/2023
@author sebastian
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats
import scipy.signal as signal
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def fourier_score(obs, obs_points, frequency_min=20e6, frequency_max=200e6):
    n = 1000
    frequencies = np.linspace(frequency_min, frequency_max, n)
    interval = (frequency_max-frequency_min)/n
    time_span = obs_points.max()-obs_points.min()
    ft = _dtft(obs, time_span, frequencies)
    arg = np.argmax(ft)
    f = frequencies[arg]
    height = ft[arg]
    widths = signal.peak_widths(ft, [arg], rel_height=0.5)
    width = widths[0][0]*interval
    half_height = widths[1]
    t2 = 1.0/(2.0*width)
    test = stats.shapiro(ft)
    noise = True if (height*0.5) < np.mean(ft)+2*np.std(ft) or math.isclose(width, 0.0) else False
    return f, t2, height, half_height, noise, test.pvalue, ft, frequencies

def _frequencies_to_periods(time_span, frequencies):
    return time_span * frequencies

def _dtft(Z, time_span, frequencies, remove_background=True):
    periods = _frequencies_to_periods(time_span, frequencies)
    Z = Z.copy()
    mean = np.nanmean(Z)

    Z[np.isnan(Z)] = mean

    if remove_background:
        gf_X = gaussian_filter(np.real(Z), sigma=5)
        gf_Y = gaussian_filter(np.imag(Z), sigma=5)
        Z = Z - (gf_X + 1j * gf_Y)

    n_0 = np.arange(0, Z.shape[0]) / Z.shape[0]
    z_0 = np.exp(-2j * np.pi * periods[:, np.newaxis] * n_0[np.newaxis, :])

    data = np.einsum('a, ba -> b', Z, z_0, optimize='greedy')
    return np.abs(data) / Z.size
"""
Created on 22/03/2023
@author sebastian
"""

from jean2.inference.fourier_utils import fourier_score

def quality_factor(measurement):
    f, t2, height, half_height, noise, pvalue, ft, frequencies = fourier_score(measurement.y, measurement.x, frequency_min=1, frequency_max=10)
    score = 0.0 if noise else 2*f*t2
    return score
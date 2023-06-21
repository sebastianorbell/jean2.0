import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import peak_widths, find_peaks, peak_prominences

def lorentzian(x, offset, amplitude, x0, gamma):
    numerator = (gamma / np.pi)
    denominator = ((x - x0)**2 + gamma**2)
    l = offset + amplitude*(numerator / denominator)
    return l

def normalise_signal_peak(y):
    peaks, peak_properties = find_peaks(y, prominence=[None, None], width=[None, None])
    troughs, trough_properties = find_peaks(-y, prominence=[None, None], width=[None, None])

    # Get the index of the peak with maximum prominence
    peak_max_prominence = np.max(peak_properties["prominences"])
    trough_max_prominence = np.max(trough_properties["prominences"])

    if trough_max_prominence > peak_max_prominence:
        y=-y
        arg_max_prominence = troughs[np.argmax(trough_properties["prominences"])]
        prominence = trough_max_prominence
        width = trough_properties['widths'][np.argmax(trough_properties["prominences"])]
    else:
        arg_max_prominence = peaks[np.argmax(peak_properties["prominences"])]
        prominence = peak_max_prominence
        width = peak_properties['widths'][np.argmax(peak_properties["prominences"])]
    y = (y-y.min())/(y.max()-y.min())
    return y, arg_max_prominence, prominence, width

def fit_lorentzian(x, y, plot=False):
    y, arg_max_prominence, prominence, width = normalise_signal_peak(y)
    offset_est = y.mean()
    amplitude_est = (y.max() - y.min())
    x0_est = x[arg_max_prominence]
    gamma_est = 0.5*width*((x.max() - x.min())/x.__len__())

    offset_bounds = (y.min(), y.max())
    amplitude_bounds = (0., amplitude_est*1.5)
    x0_bounds = (x.min(), x.max())
    gamma_bounds = (gamma_est*0.5,gamma_est*2.)
    bounds = list(zip(offset_bounds, amplitude_bounds, x0_bounds, gamma_bounds))

    # the initial guess
    p0 = [offset_est, amplitude_est, x0_est, gamma_est]
    popt, pcov = curve_fit(lorentzian, x, y, p0, bounds=bounds)

    # extracting the important parameters from the fit
    offset_pred, amplitude_pred, x0_pred, gamma_pred = popt

    # computing the predicted signal
    y_pred = lorentzian(x, *popt)

    # computing the signal-to-noise ratio
    least_squares_noise = np.sqrt(np.mean((y - y_pred) ** 2))
    signal_to_noise = amplitude_pred / least_squares_noise

    if plot:
        plt.plot(x, y_pred)
        plt.scatter(x, y)
        plt.axvline(x0_pred, linestyle='dashed')
        plt.title(f'SNR={signal_to_noise}')
        plt.show()

    return x0_pred, gamma_pred, signal_to_noise
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def decaying_cosine(t, frequency, amplitude, phase, offset, decay):
    """
    compute the decaying cosine
    @param t:
    @param frequency:
    @param amplitude:
    @param phase:
    @param offset:
    @param decay:
    @return:
    """
    return amplitude * np.exp(-decay * t) * np.cos(2 * np.pi * frequency * t + phase) + offset


def decaying_envelope_upper(t, frequency, amplitude, phase, offset, decay):
    """
    compute the decaying envelope upper side
    @param t:
    @param frequency:
    @param amplitude:
    @param phase:
    @param offset:
    @param decay:
    @return:
    """
    return amplitude * np.exp(-decay * t) + offset


def decaying_envelope_lower(t, frequency, amplitude, phase, offset, decay):
    """
    compute the decaying envelope lower side
    @param t:
    @param frequency:
    @param amplitude:
    @param phase:
    @param offset:
    @param decay:
    @return:
    """
    return -amplitude * np.exp(-decay * t) + offset


def dtft(Z, time_span, frequencies, subtract_mean=True):
    """
    Compute the discrete time fourier transform
    @param Z:
    @param time_span:
    @param frequencies:
    @param subtract_mean:
    @return:
    """
    periods = time_span * frequencies

    if subtract_mean:
        Z = Z - Z.mean()

    n_0 = np.arange(0, Z.shape[0]) / Z.shape[0]
    z_0 = np.exp(-2j * np.pi * periods[:, np.newaxis] * n_0[np.newaxis, :])

    data = np.einsum('a, ba -> b', Z, z_0, optimize='greedy')
    return np.abs(data) / Z.size

def check_frequency_components(f_spectrum):
    peaks, peak_properties = find_peaks(f_spectrum, prominence=[None, None])
    prominences = peak_properties['prominences']
    if prominences.__len__() == 1:
        sorted_indexes=[0]
        return peaks, prominences, sorted_indexes
    else:
        sorted_indexes = np.argsort(prominences)
        return peaks, prominences, sorted_indexes

def fit_decaying_cosine(t, y, f_min=None, f_max=None, plot=True):
    """
    Fit a decaying sine to the data
    :param t: the time points in s
    :param y: the measured demodulate signal
    :param plot: a bool whether to plot the fit
    :return: f_pred, T2_pred, signal_to_noise the predicted frequency and T2 in MHz and us respectively
    """

    y = (y - y.min()) / (y.max() - y.min())

    # the time span for the measurement
    time_span = t.max() - t.min()
    delta_t = time_span / len(t)
    f_niquist = 1 / (2 * delta_t)

    # the frequencies to computer the fourier transform for
    if not f_min:
        f_min = 0.
    if not f_max:
        f_max = f_niquist

    frequencies = np.arange(f_min, f_max, 1e4)

    # computing the fourier transform
    ft = dtft(y, time_span, frequencies)

    # finding the maximum of the fourier transform
    arg = np.argmax(ft)
    f_est = frequencies[arg]
    amplitude_est = (y.max() - y.min()) / 2.
    phase_est = np.pi
    offset_est = y.mean()
    T2_est = 1e6

    # the bounds for the fit
    f_bounds = (f_min, f_max)
    amplitude_bounds = (0, 2 * amplitude_est)
    phase_bounds = (0, 2 * np.pi)
    offset_bounds = (y.min(), y.max())
    T2_bounds = (0, 10e6)
    bounds = list(zip(f_bounds, amplitude_bounds, phase_bounds, offset_bounds, T2_bounds))

    # the initial guess
    p0 = [f_est, amplitude_est, phase_est, offset_est, T2_est]
    popt, pcov = curve_fit(decaying_cosine, t, y, p0, bounds=bounds)

    # extracting the important parameters from the fit
    f_pred, amplitude_pred, phase_pred, offset_pred, gamma_T2_pred = popt
    T2_pred = 1 / gamma_T2_pred

    # changing units
    T2_us = T2_pred * 1e6
    f_MHz = f_pred / 1e6

    # computing the predicted signal
    y_pred = decaying_cosine(t, *popt)
    ft_pred = dtft(y_pred, time_span, frequencies)

    # computing the signal-to-noise ratio
    least_squares_noise = np.sqrt(np.mean((y - y_pred) ** 2))
    signal_to_noise = amplitude_pred / least_squares_noise

    # a block to plot the fit and the fourier transform
    if plot:
        print(f"signal to noise: {signal_to_noise}")
        print(f"Nyquist frequency: {f_niquist / 1e6:.2f}MHz")

        t_plot = t * 1e9
        frequencies_plot = frequencies / 1e6

        fig, axs = plt.subplots(2)
        fig.suptitle(f'SNR={signal_to_noise}')
        axs[0].plot(t_plot, y, label='data', marker='o', color='b')
        axs[0].plot(t_plot, y_pred, label='fit', color='r')
        axs[0].plot(t_plot, decaying_envelope_upper(t, *popt), color='k', alpha=0.5, linestyle='--')
        axs[0].plot(t_plot, decaying_envelope_lower(t, *popt), color='k', alpha=0.5, linestyle='--')
        axs[0].axhline(offset_pred, color='k', alpha=0.5, linestyle='--')

        axs[0].legend()

        axs[0].set_xlabel('Time (ns)')
        axs[0].set_ylabel('Amplitude (V)')

        axs[1].plot(frequencies_plot, ft, label='data')
        axs[1].plot(frequencies_plot, ft_pred, label='fit', color='r')

        axs[1].set_xlabel('Frequency (MHz)')
        axs[1].set_ylabel('Fourier Amplitude (V)')
        axs[1].axvline(f_est / 1e6, color='r', label='prior frequency')
        axs[1].axvline(f_pred / 1e6, color='g', label='posterior frequency')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    return f_MHz, T2_us, signal_to_noise, ft

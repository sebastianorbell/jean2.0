from doNdAWG import (do1dAWG,
                     do2dAWG,
                     init_Rabi)
from src.inference import normalise, high_pass, fit_decaying_cosine, PCA, fit_lorentzian, check_frequency_components
import numpy as np
import matplotlib.pyplot as plt
from qcodes.utils.dataset.doNd import do1d, do2d, do0d
from scipy.signal import find_peaks
from src.utils import Results

def measure_rabi_2d(vals, set_vals, start, stop, steps, ips, pp, awg, cpg, LIX, LIY, plot=True):

    set_vals(*vals)

    resonant_field, field_sweep_snr, field_data, current = measure_resonant_peak(start, stop, steps, ips, pp, awg, cpg, LIX, LIY)

    if field_sweep_snr < .5:
        return Results(rabi_frequency=0.0,
                       resonant_field=0.0,
                       b_field=None,
                       t_burst=None,
                       pca=None,
                       frequency=None,
                       snr=None,
                       field_data=field_data,
                       current=current,
                       threshold=1.5)

    pp = rabi_pulsing(pp)
    delta_field = 0.03
    ips.field_setpoint_wait(resonant_field-delta_field)
    data = do2dAWG("Rabi", ips.field_setpoint_wait, resonant_field-delta_field, resonant_field + delta_field, 15, 4,
                   "t_burst", 0e-9, 45e-9, 45, 3,
                   LIX, LIY, pp=pp, awg=awg, cgp=cpg, show_progress=False, show_pulse=False)
    xarray = data[0].to_xarray_dataset()
    x = xarray['LIX'].to_numpy()
    y = xarray['LIY'].to_numpy()
    t_burst = xarray['t_burst'].to_numpy()
    b_field = xarray['IPS_field_setpoint_wait'].to_numpy()
    f_min = 50e6
    x = normalise(high_pass(x, times=t_burst, f_min=f_min))
    y = normalise(high_pass(y, times=t_burst, f_min=f_min))
    pca = high_pass(PCA(x, y), times=t_burst, f_min=f_min)
    snr = []
    frequency=[]
    spectrum = []
    for z in pca:
        f, t2, s, ft = fit_decaying_cosine(t_burst, z, f_min=f_min, f_max=250e6, plot=False)
        frequency.append(f)
        snr.append(s)
        spectrum.append(ft)
    snr = np.array(snr)
    frequency = np.array(frequency)

    threshold=1.5

    rabi_frequency = 0.0 if np.all(snr < threshold) else np.min(frequency[snr>threshold])
    resonant_field = 0.0 if np.all(snr < threshold) else b_field[np.argmin(frequency[snr>threshold])]

    if rabi_frequency != 0.0:
        ft = spectrum[np.argmin(frequency[snr>threshold])]
        peaks, prominences, sorted_indexes = check_frequency_components(ft)
        if prominences[-1] <= prominences[-2]*2.:
            rabi_frequency = 0.0
            resonant_field = 0.0

    if plot:
        if np.all(snr < threshold):
            pass
        else:
            f, t2, s ,_ = fit_decaying_cosine(t_burst, pca[np.argmin(frequency[snr>threshold])], f_min=f_min, f_max=250e6, plot=True)

    return Results(rabi_frequency=rabi_frequency ,
                   resonant_field=resonant_field,
                   b_field=b_field,
                   t_burst=t_burst,
                   pca=pca,
                   frequency=np.array(frequency),
                   snr=np.array(snr),
                   field_data=field_data,
                   current=current,
                   threshold=threshold)

def measure_resonant_peak(start, stop, steps, ips, pp, awg, cpg, LIX, LIY, plot=True):
    pp = continuous_pulsing(pp)
    init_Rabi(pp, awg, cpg, plot=False)
    ips.field_setpoint_wait(start)
    data = do1d(ips.field_setpoint_wait, start, stop, steps, 3, LIX, LIY, show_progress=False)
    xr_data = data[0].to_xarray_dataset()
    xdata = xr_data['LIX'].to_numpy()
    ydata = xr_data['LIY'].to_numpy()
    field_data = xr_data['IPS_field_setpoint_wait'].to_numpy()

    pca = PCA(xdata, ydata)
    pca = normalise(pca)

    x0, gamma, snr = fit_lorentzian(field_data, pca, plot=plot) # vrp = -0.00388 vlp = 0.12925

    return x0, snr, field_data, pca


def meta_score_function(vrp=None, vlp=None, parameter=None, measurement=None):
    VRP(vrp)
    VLP(vlp)
    jean = Jean(parameters=parameter,
                     n_calls=30,
                     n_initial_points=20,
                     score_function=rabi_frequency,
                     measurement=measurement,
                     database=Database(inner_loop_dir),
                     plot=True)

    res = jean()

    optimal_measurement_results = jean.measurement_results[tuple(res.x)]
    t2 = optimal_measurement_results.t2
    snr = optimal_measurement_results.snr
    signal = optimal_measurement_results.signal
    t_burst = optimal_measurement_results.t_burst

    return Results(sweep_value=res.x,
                   sweep=optimal_measurement_results.sweep,
                   frequency=-res.fun,
                   t2=t2,
                   snr=snr,
                   res=res,
                   signal=signal,
                   t_burst=t_burst,
                   measurements=jean.measurement_results)


def rabi_1d(value, run_fn, sweep='freq'):
    if sweep=='freq':
        VS_freq(value*1e9)
    if sweep=='field':
        ips.field_setpoint_wait(round(b_field, 4))
        while round(ips.field(), 4)!=round(b_field, 4):
            pass
    # VS_freq(freq*1.e9)
    res = run_fn()
    xr_data = res[0].to_xarray_dataset()
    x = xr_data['LIX'].to_numpy()[discard:]
    y = xr_data['LIY'].to_numpy()[discard:]
    t_burst = xr_data['t_burst'].to_numpy()[discard:]
    f_min = 75e6
    f_max = 250e6
    x = normalise(high_pass(x, times=t_burst, f_min=f_min))
    y = normalise(high_pass(y, times=t_burst, f_min=f_min))
    pca = high_pass(PCA(x, y), times=t_burst, f_min=f_min)
    f_pred, T2_pred, signal_to_noise = fit_decaying_cosine(t_burst, pca, f_min=f_min*1.5, f_max=f_max, plot=False)
    return Results(t2=T2_pred, f=f_pred, snr=signal_to_noise, signal=pca, t_burst=t_burst, sweep_value=value, sweep=sweep)

def rabi_pulsing(pp):
    pp.C_ampl = -0.025
    pp.t_RO = 50e-9
    pp.t_CB = 50e-9
    pp.t_ramp = 4e-9
    pp.t_burst = 4e-9
    pp.IQ_delay = 19e-9
    pp.I_ampl = 0.3
    pp.Q_ampl = 0.3
    return pp

def continuous_pulsing(pp):
    pp.C_ampl = -0.025
    pp.t_RO = 20e-9
    pp.t_CB = 20e-9
    pp.t_ramp = 5e-9
    pp.t_burst = 5e-9
    pp.IQ_delay = 19e-9  # delay calibrated with scope
    pp.I_ampl = 0.3
    pp.Q_ampl = 0.3
    return pp
import os
import unittest
from src import Database, Jean, Experiment_parameter, fit_decaying_cosine, Results
from src.utils.database.database import create_directory_structure
from src.inference import PCA
from init_basel import initialise_experiment, rabi_pulsing
from qcodes.utils.dataset.doNd import do1d, do2d, do0d
import matplotlib.pyplot as plt

from doNdAWG import (do1dAWG,
                     do2dAWG,
                     init_Rabi)

import numpy as np

data_dir = os.getcwd() + "/data"
experiment_name = 'preliminary_rabi'
experiment_dir = create_directory_structure(f'{data_dir}/{experiment_name}/')
inner_loop_dir = create_directory_structure(f'{data_dir}/{experiment_name}/inner_loop')
meta_data_dir = create_directory_structure(f'{data_dir}/{experiment_name}/meta_data')

drive_frequency_bounds = {'drive_freq': (2.8e9, 3.2e9)}
drive_frequency_parameter = Experiment_parameter(drive_frequency_bounds)

def rabi_1d(freq, pp=None, awg=None, cpg=None):
    # ips.run_to_field_wait(b_field)
    VS_freq(freq)
    res = do1dAWG("Rabi", "t_burst", 0., 60e-9, 60, 2.1, LIX, LIY, pp=pp, awg=awg, cgp=cpg, show_progress=False, show_pulse=False)
    xr_data = res[0].to_xarray_dataset()
    x = xr_data['LIX'].to_numpy()[10:]
    y = xr_data['LIY'].to_numpy()[10:]
    pca = PCA((x-np.mean(x))/np.std(x),(y-np.mean(y))/np.std(y))
    t_burst = xr_data['t_burst'].to_numpy()[10:]
    f_pred, T2_pred, noise_to_signal = fit_decaying_cosine(t_burst, pca, f_min=50.e6, f_max=200.e6, plot=False)
    return Results(t2=T2_pred, f=f_pred, snr=1./noise_to_signal, signal=pca, t_burst=t_burst, drive_freq=freq)

def rabi_frequency(results):
    if results.snr < 1.:
        return 0.
    else:
        return -1./results.f

def meta_score_function(vrp=None, vlp=None, parameter=None, measurement=None):
    VRP(vrp)
    VLP(vlp)
    jean = Jean(parameters=parameter,
                     n_calls=20,
                     n_initial_points=10,
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

    return Results(drive_freq=res.x,
                   frequency=-res.fun,
                   t2=t2,
                   snr=snr,
                   signal=signal,
                   t_burst=t_burst,
                   measurements=jean.measurement_results)

def quality_factor(results):
    return -results.frequency * results.t2

gate_bounds = {'vrp': (-0.013, 0.012), 'vlp': (0.103, 0.128)}
meta_parameter = Experiment_parameter(bounds=gate_bounds)

awg, pp, DAQ, mfli, VS, VS_freq, VS_phase, VS_pwr, VS_pulse_on, VS_pulse_off, VS_IQ_on, VS_IQ_off, VS_status, LIXY, LIXYRPhi, LIX, LIY, LIR, LIPhi, LIPhaseAdjust, LIfreq, LITC, ISD, DAQ, VS, VM, VL, VLP, VR, VRP = initialise_experiment()
LITC(2)

known_rabi_spot = {
    'VSD':0.519,
    'VR':1.02,
    'VL':1.35,
    'VLP':0.1185,
    'VRP':7.97e-5,
    'VM':0.66
}

pp = rabi_pulsing(pp)
VLP(known_rabi_spot['VLP'])
VRP(known_rabi_spot['VRP'])

rabi_1d_measurement = lambda drive_freq=None: rabi_1d(drive_freq, pp=pp, awg=awg, cpg=VRP)

meta_jean = Jean(parameters=meta_parameter,
                 n_calls=10,
                 n_initial_points=5,
                 score_function=quality_factor,
                 measurement=lambda vrp, vlp : meta_score_function(vrp=vrp, vlp=vlp,
                                                                   parameter=drive_frequency_parameter,
                                                                   measurement=rabi_1d_measurement),
                 database=Database(experiment_dir),
                 plot=True)

res = meta_score_function(known_rabi_spot['VRP'], known_rabi_spot['VLP'],
                            parameter=drive_frequency_parameter,
                          measurement=rabi_1d_measurement)

VS_freq(res.drive_freq[0])
VS_freq(3.03e9)
measurement_res = do1dAWG("Rabi", "t_burst", 0., 60e-9, 60, 3, LIX, LIY,LIPhi, LIR, pp=pp, awg=awg, cgp=VRP, show_progress=False, show_pulse=False)
xr_data = measurement_res[0].to_xarray_dataset()
x = xr_data['LIX'].to_numpy()[10:]
y = xr_data['LIY'].to_numpy()[10:]
x_n = (x - np.mean(x)) / np.std(x)
y_n = (y - np.mean(y)) / np.std(y)
pca = PCA(x_n, y_n)
t_burst = xr_data['t_burst'].to_numpy()[10:]
f_pred, T2_pred, noise_to_signal = fit_decaying_cosine(t_burst, pca, f_min=50e6, f_max=200e6, plot=True)

plt.plot(t_burst, x_n, label='x')
plt.plot(t_burst, y_n, label='y')
plt.plot(t_burst, pca, '--', label='pca')
plt.xlabel('t-burst')
plt.legend()
plt.show()

# LITC(0.5)
# scan = do2d(VLP, gate_bounds['VLP'][0], gate_bounds['VLP'][1], 50, 1, VRP, gate_bounds['VRP'][0], gate_bounds['VRP'][1], 50, 0.51)

# LITC(2)
# res = meta_jean()

import os
import unittest
from src import Database, Jean, Experiment_parameter, fit_decaying_cosine, Results
from src.utils.database.database import create_directory_structure
from src.inference import PCA, normalise, high_pass, low_pass
from init_basel import *
from utils_basel import *
from qcodes.utils.dataset.doNd import do1d, do2d, do0d
import matplotlib.pyplot as plt

from doNdAWG import (do1dAWG,
                     do2dAWG,
                     init_Rabi,
                     load_do1dAWG,
                     run_do1dAWG)

import numpy as np

data_dir = os.getcwd() + "/data"
experiment_name = 'four_gates'
experiment_dir = create_directory_structure(f'{data_dir}/{experiment_name}/')
meta_data_dir = create_directory_structure(f'{data_dir}/{experiment_name}/meta_data')

drive_frequency_bounds = {'sweep_value': (2.45, 3.35)}
drive_frequency_parameter = Experiment_parameter(drive_frequency_bounds)

b_field_bounds = {'sweep_value':(0.28, 0.33)}
b_field_parameter = Experiment_parameter(b_field_bounds)

def awg_fns(*args, **kwargs):
    load = lambda: load_do1dAWG(*args, **kwargs)
    run = lambda: run_do1dAWG(*args, **kwargs)
    return load, run

def rabi_frequency(results):
    if results.snr < 1.25 :
        return 0.
    else:
        return -1./results.f
def chevron_score_function(results):
    return  -results.rabi_frequency

ips, awg, pp, DAQ, mfli, VS, VS_freq, VS_phase, VS_pwr, VS_pulse_on, VS_pulse_off, VS_IQ_on, VS_IQ_off, VS_status, LIXY, LIXYRPhi, LIX, LIY, LIR, LIPhi, LIPhaseAdjust, LIfreq, LITC, ISD, DAQ, VS, VM, VL, VLP, VR, VRP, VSD = initialise_experiment()

known_rabi_spot = {
    'VL': 1.32,
    'VR': 1.019999,
    'VM': 0.659999,
    'VSD': 0.538721,
    'VLP': 0.13,
    'VRP': -0.00731578947368422
}

def gate_voltages():
    print(f'VL: {VL()},')
    print(f'VR: {VR()},')
    print(f'VM: {VM()},')
    print(f'VSD: {VSD()},')
    print(f'VLP: {VLP()},')
    print(f'VRP: {VRP()}')

pp = rabi_pulsing(pp)

rabi_1d_measurement = lambda sweep_value=None: rabi_1d(sweep_value, run, sweep='freq')

LITC(2)
def set_vals(vrp, vlp):#, vr, vl):
    VRP(vrp)
    VLP(vlp)
    # VR(vl)
    # VL(vl)

vals = [known_rabi_spot['VRP'], known_rabi_spot['VLP']]

measurement_fn = lambda x: measure_rabi_2d(x, set_vals, 0.2, 0.4, 50, ips, pp, awg, VRP, LIX, LIY)
gate_bounds = {
    'vrp': (-0.020, 0.020),
   'vlp': (0.115, 0.145),
   # 'vr': (0.8, 1.2),
   # 'vl': (1.1, 1.5)
}

gate_parameter = Experiment_parameter(bounds=gate_bounds)
experiment_database = Database(experiment_dir)
jean = Jean(parameters=gate_parameter,
                 n_calls=50,
                 n_initial_points=30,
                 score_function=chevron_score_function,
                 measurement=lambda **kwargs : measurement_fn(list(kwargs.values())),
                 database=experiment_database,
                 plot=True)

# rabi2d = do2dAWG("Rabi",  "t_burst", 0., 60.e-9, 60, 2.1, VS_freq, drive_frequency_bounds['drive_freq'][0]*1e9, drive_frequency_bounds['drive_freq'][1]*1e9, 20, 2.1, ISD, LIX, LIY, LIR, pp=pp, awg=awg, cgp=VRP, show_progress=True)

LITC(0.5)
scan = do2d(VLP, gate_bounds['VLP'][0], gate_bounds['VLP'][1], 50, 1, VRP, gate_bounds['VRP'][0], gate_bounds['VRP'][1], 50, 0.51, ISD, LIX, LIY, LIR)

LITC(4.)
res = jean()

# data_list = []
# for litc in litc_list:
#     LITC(litc)
#     data = do2dAWG("Rabi", ips.field_setpoint_wait, 0.3, 0.33, 21, litc * 1.2,
#                    "t_burst", 0e-9, 45e-9, 45, litc * 1.1,
#                    LIX, LIY, pp=pp, awg=awg, cgp=VRP, show_progress=False, show_pulse=False)
#     data_list.append(data)

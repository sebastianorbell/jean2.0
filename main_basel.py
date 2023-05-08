import os
import unittest
from src import Database, Jean, Experiment_parameter, fit_decaying_cosine, Results
from src.utils.database.database import create_directory_structure
import numpy as np

data_dir = os.getcwd() + "/data"
experiment_name = 'preliminary_rabi'
experiment_dir = create_directory_structure(f'{data_dir}/{experiment_name}/')
inner_loop_dir = create_directory_structure(f'{data_dir}/{experiment_name}/inner_loop')
meta_data_dir = create_directory_structure(f'{data_dir}/{experiment_name}/meta_data')

field_bounds = {'b_field': (20., 30.)}
magnetic_field_parameter = Experiment_parameter(field_bounds)

def rabi_1d(b):
    ips.run_to_field_wait(b)
    res = do1dAWG("Rabi", "t_burst", 0., 10e-9, 50, .5, ISD, LIX, LIY, LIR, pp=pp, awg=awg, cgp=VRP, show_progress=True)
    return res

def rabi_1d(b_field=b_field):
    ips.run_to_field_wait(b_field)
    res = do1dAWG("Rabi", "t_burst", 0., 10e-9, 50, .5, ISD, LIX, LIY, LIR, pp=pp, awg=awg, cgp=VRP, show_progress=True)
    f_pred, T2_pred, noise_to_signal = fit_decaying_cosine(t, y)
    return Results(t2=T2_pred, f=f_pred)

def rabi_frequency(results):
    return results.f

def meta_score_function(vrp, vlp):
    VRP(vrp)
    VLP(vlp)
    jean = Jean(parameters=magnetic_field_parameter,
                     n_calls=10,
                     n_initial_points=5,
                     score_function=rabi_frequency,
                     measurement=rabi_1d,
                     database=Database(inner_loop_dir),
                     plot=False)

    res = jean()

    optimal_measurement_results = jean.measurement_results[tuple(res.x)]

    return Results(field=res.x, frequency=res.fun, t2=optimal_measurement_results.t2)

def quality_factor(results):
    return results.frequency / results.t2

gate_bounds = {'vrp': (-0.013, 0.012), 'vlp': (0.103, 0.128)}
meta_parameter = Experiment_parameter(bounds=gate_bounds)

meta_jean = Jean(parameters=meta_parameter,
                 n_calls=10,
                 n_initial_points=5,
                 score_function=quality_factor,
                 measurement=meta_score_function,
                 database=Database(experiment_dir),
                 plot=True)

res = meta_jean()
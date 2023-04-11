"""
Created on 22/03/2023
@author sebastian
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skopt import gp_minimize
from skopt.plots import plot_evaluations, plot_objective, plot_convergence, plot_gaussian_process
from time import time
from src.utils.database import Database
from src.utils.Dummy_parameter import Dummy_parameter

class Jean:
    def __init__(self,
                 parameters,
                 bounds,
                 n_calls,
                 n_initial_points,
                 score_function,
                 measurement,
                 database,
                 x0=None,
                 y0=None,
                 plot=True):
        """

        Main class which performs optimisation and saves the results to a database.

        :param parameters: List of qgor parameters.
        :param bounds: List of tuples indicating the parameter bounds.
        :param n_calls: Number of calls of the objective function.
        :param n_initial_points: Number of initial points.
        :param score_function: Function which takes the measurement results as an argument and returns a float.
        :param measurement: Function which takes the query values and returns measurement data.
        :param database: Database for the optimisation.
        """
        self.parameters = parameters
        self.bounds = bounds
        assert self.parameters.__len__() == self.bounds.__len__(), \
            f'bounds {self.bounds.__len__()} must be the same length as parameters {self.parameters.__len__()}'
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        self.names = [parameter.name for parameter in parameters]
        self.score_function = score_function
        self.measurement = measurement
        self.times = []
        self.measurement_results = []
        self.database = database
        self.plot = plot
        self.x0 = x0
        self.y0 = y0

    def __call__(self):
        """
        Run the optimisation.
        @return:
        """
        res = gp_minimize(
            self._objective_function,
            self.bounds,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            x0=self.x0,
            y0=self.y0
        )

        self.database.save_dataset(res, self)
        if self.plot:
            self._plot(res)
        return res

    def _plot(self, res):
        """
        Plot the results.
        @param res:
        @return:
        """
        plot_evaluations(res)
        plt.show()

        plot_objective(res)
        plt.show()

        plot_convergence(res)
        plt.show()

        if res.space.n_dims == 1:
            plot_gaussian_process(res)
            plt.show()

    def _objective_function(self, values):
        """
        Objective function which is minimised by the optimiser.
        @param values:
        @return:
        """
        assert values.__len__() == self.parameters.__len__(), \
            f'values {values.__len__()} must be the same length as parameters {self.parameters.__len__()}'
        self._timeit()
        for value, parameter in zip(values, self.parameters):
            parameter.set(value)
        measurement_result = self._measurement()
        score = self.score_function(measurement_result)
        return score

    def _measurement(self):
        """
        Perform a measurement.
        @return:
        """
        measurement_result = self.measurement()
        self.measurement_results.append(measurement_result)
        return measurement_result

    def _timeit(self):
        """
        Save the current time.
        @return:
        """
        self.times.append(time())


if __name__ == '__main__':
    def measurement(value):
        class Measurement:
            def __init__(self, value):
                self.x = np.square(value - 0.2)  # + np.random.rand(1)

        m = Measurement(value)
        return m


    experiment_directory = os.getcwd() + '/../data'
    sys.path.append(experiment_directory)
    database = Database(experiment_directory)
    database_id = database.next_id()
    sub_dir = f'{experiment_directory}/{database_id}'
    sys.path.append(sub_dir)
    sub_database = Database(sub_dir)

    dummy_measurement_parameter = Dummy_parameter('measurement', init_value=0, measurement=measurement)

    sub_jean = Jean(
        parameters=[dummy_measurement_parameter],
        bounds=[(0., 1.)],
        n_calls=20,
        n_initial_points=10,
        score_function=lambda x: x.x,
        measurement=dummy_measurement_parameter.measurement,
        database=sub_database,
        plot=False
    )

    sub_res = sub_jean()

    dummy2 = Dummy_parameter('dummy2')
    dummy1 = Dummy_parameter('dummy1')

    jean = Jean(
        parameters=[dummy1, dummy2],
        bounds=[(0., 1.), (0., 1.)],
        n_calls=20,
        n_initial_points=10,
        score_function=lambda x: x.fun,
        measurement=sub_jean,
        database=database,
        plot=True
    )

    res = jean()

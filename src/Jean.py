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
from .utils.database import Database
from .utils.Experiment_parameter import Experiment_parameter

class Jean:
    def __init__(self,
                 parameters,
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
        :param n_calls: Number of calls of the objective function.
        :param n_initial_points: Number of initial points.
        :param score_function: Function which takes the measurement results as an argument and returns a float.
        :param measurement: Function which takes the query values and returns measurement data.
        :param database: Database for the optimisation.
        """
        self.parameters = parameters
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
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
            self.parameters.bounds,
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
        self._timeit()
        self.parameters.set(values)
        measurement_result = self._measurement()
        score = self.score_function(measurement_result)
        return score

    def _measurement(self):
        """
        Perform a measurement.
        @return:
        """
        measurement_result = self.measurement(**self.parameters.get())
        self.measurement_results.append(measurement_result)
        return measurement_result

    def _timeit(self):
        """
        Save the current time.
        @return:
        """
        self.times.append(time())



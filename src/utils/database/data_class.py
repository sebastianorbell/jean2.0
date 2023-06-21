"""
Created on 09/12/2021
@author jdh
"""

from dataclasses import dataclass
import numpy as np
import argparse
import pickle


@dataclass
class DataClass:
    """
    Class for saving optimisation data
    """
    date: str
    optimal_value: float
    optimal_point: np.ndarray
    query_values: list
    query_points: list
    parameter_names: list
    bounds: list
    n_calls: int
    n_initial_points: int
    times: list
    measurement_results: list
    id: str

    def save(self, filename):
        """
        Pickle the optimisation data and save it to filename
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

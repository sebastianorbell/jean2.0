"""
Created on 09/12/2021
@author jdh
"""

import os
from pathlib import Path
import pickle
from datetime import datetime
import numpy as np

from .data_class import DataClass
from .helpers import find_min_and_max_id


def create_directory(folder_path):
    """
    a function to create a directory and folder path
    @param folder_path: where to create the directory
    """
    if not os.path.exists(folder_path): os.mkdir(folder_path)


def create_directory_structure(path):
    """
    a function to create the directory structure. Such that if you call it to create ./a/b/c. And ./a/ does not exist
    it will create it
    @param path:
    """
    for path_parent in reversed(Path(path).parents):
        create_directory(path_parent)
    create_directory(path)
    return path


class Database:
    """
    Create a database for a single experiment run.
    """

    def __init__(self, experiment_directory, **kwargs):
        self.experiment_directory = Path(experiment_directory)
        self.kwargs = kwargs
        self.ids_for_optimiser = []
        create_directory_structure(self.experiment_directory)
        print("created database at {}".format(self.experiment_directory))

    def save_dataset(self, res, jean):
        id = str(self.next_id())

        dataclass = DataClass(
            date=datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            optimal_value=res.fun,
            optimal_point=res.x,
            query_values=res.func_vals,
            query_points=res.x_iters,
            parameter_names=jean.parameters.names,
            bounds=jean.parameters.bounds,
            n_calls=jean.n_calls,
            n_initial_points=jean.n_initial_points,
            times=jean.times,
            measurement_results=jean.measurement_results,
            id=id
        )

        filename = id + '.pkl'

        # this just saves the directory - need the filename now
        dataclass.save(self.experiment_directory / Path(filename))

        return id

    def last_id(self):
        _, last_id = find_min_and_max_id(self.experiment_directory)
        if last_id is None:
            last_id = 0
        return last_id

    def next_id(self):
        return self.last_id() + 1

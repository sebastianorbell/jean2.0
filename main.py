"""
Created on 22/03/2023
@author sebastian
"""
import numpy as np
import os
import sys
from jean2.utils.database import Database
from jean2.main import Dummy_parameter
from jean2.main import Jean

if __name__=='__main__':
    def measurement(value):
        class Measurement:
            def __init__(self, value):
                self.x = np.square(value-0.2) #+ np.random.rand(1)
        m = Measurement(value)
        return m

    experiment_directory = os.getcwd()+'/data'
    sys.path.append(experiment_directory)
    database = Database(experiment_directory)
    database_id = database.next_id()
    sub_dir = f'{experiment_directory}/{database_id}'
    sys.path.append(sub_dir)
    sub_database = Database(sub_dir)

    dummy_measurement_parameter = Dummy_parameter('measurement', init_value=0, measurement=measurement)

    sub_jean = Jean(
                     parameters=[dummy_measurement_parameter],
                     bounds=[(0.,1.)],
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
                     bounds=[(0.,1.), (0.,1.)],
                     n_calls=20,
                     n_initial_points=10,
                     score_function=lambda x: x.fun,
                     measurement=sub_jean,
                     database=database,
                    plot=True
    )

    res = jean()
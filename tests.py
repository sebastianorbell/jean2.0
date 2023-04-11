import os
import unittest
from src import Database, Jean, Dummy_parameter, fit_decaying_cosine
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_create_database(self, directory=os.getcwd()+"/data"):
        self.database = Database(directory)
        self.assertTrue(self.database)
    def test_create_dummy_parameter(self):
        def auto_characterization(x):
            return Results(t2=1/x, larmor=1/x)

        self.parameter = Dummy_parameter('epsilon',
                                         init_value=0.0,
                                         measurement=lambda x: auto_characterization(x))
        self.assertTrue(self.parameter)
    def test_create_jean(self):
        self.test_create_database()
        self.test_create_dummy_parameter()
        self.jean = Jean(parameters=[self.parameter],
                         bounds=[(15.,25.)],
                         n_calls=10,
                         n_initial_points=5,
                         score_function=lambda x: 1/x.t2,
                         measurement=self.parameter.measurement,
                         database=self.database,
                         plot=True,
                         x0=[[23.295720250820295], [24.52004700612897], [19.018953133673882]],
                         y0=1/np.array([0.04292634, 0.04078296, 0.05257913])
                         )
        self.assertTrue(self.jean)

    def test_run_jean(self):
        self.test_create_jean()
        self.res = self.jean()
        self.assertTrue(self.res)

    def test_fit_decaying_cosine(self):
        t = np.linspace(0, 30, 100)
        y = np.cos(2 * np.pi * 10 * t + np.pi) * np.exp(-t*1e-1) + np.random.normal(0, 0.1, 100)
        f_pred, T2_pred, signal_to_noise = fit_decaying_cosine(t*1e-9, y)
        self.assertTrue(f_pred)

class Results:
    def __init__(self, t2, larmor):
        self.t2 = t2
        self.larmor=larmor

if __name__ == '__main__':
    unittest.main()

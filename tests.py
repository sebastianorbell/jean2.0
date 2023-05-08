import os
import unittest
from src import Database, Jean, Experiment_parameter, fit_decaying_cosine, Results
from src.utils.database.database import create_directory_structure
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_create_database(self, directory=os.getcwd()+"/data"):

        self.database = Database(directory)
        self.assertTrue(self.database)
    def test_create_dummy_parameter(self):

        self.parameter = Experiment_parameter(bounds={'epsilon': (15., 25.)})
        self.assertTrue(self.parameter)

    def test_create_jean(self):
        self.test_create_database()
        self.test_create_dummy_parameter()

        def auto_characterization(epsilon=None):
            return Results(t2=1/epsilon, larmor=1/epsilon)

        self.jean = Jean(parameters=self.parameter,
                         n_calls=10,
                         n_initial_points=5,
                         score_function=lambda x: 1/x.t2,
                         measurement=auto_characterization,
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

    def test_create_jean_2d(self):
        self.test_create_database()

        bounds = {'epsilon': (20., 30.), 'theta': (0, np.pi / 2)}
        self.parameter = Experiment_parameter(bounds)

        def auto_characterization(epsilon = None, theta = None):
            return Results(t2=epsilon * np.sin(theta) + 0.1, larmor=theta/epsilon)

        def score_function(results):
            return results.larmor / results.t2

        self.jean = Jean(parameters=self.parameter,
                         n_calls=10,
                         n_initial_points=5,
                         score_function=score_function,
                         measurement=auto_characterization,
                         database=self.database,
                         plot=True)

        self.res = self.jean()
        self.assertTrue(self.res)

    def test_fit_decaying_cosine(self):
        t = np.arange(0, 3000, 50) / 1e9
        f_true = 1e6  # Hz
        T2 = 1e-6  # s
        noise = 0.01  # V
        y = np.cos((2 * np.pi) * f_true * t + np.pi) * np.exp(-t / T2) + noise * np.random.randn(*t.shape)
        f_pred, T2_pred, noise_to_signal = fit_decaying_cosine(t, y)
        self.assertAlmostEqual(f_pred, f_true / 1e6, delta=1e-1)
        self.assertAlmostEqual(T2_pred, T2 * 1e6, delta=1e-1)
        self.assertAlmostEqual(noise_to_signal, noise, delta=1e-1)

    def test_nested_jean(self):
        self.test_create_database()
        data_dir = os.getcwd() + "/data"
        experiment_name = 'preliminary_rabi'
        experiment_dir = create_directory_structure(f'{data_dir}/{experiment_name}/')
        inner_loop_dir = create_directory_structure(f'{data_dir}/{experiment_name}/inner_loop')
        meta_data_dir = create_directory_structure(f'{data_dir}/{experiment_name}/meta_data')

        bounds = {'epsilon': (20., 30.), 'theta': (0, np.pi / 2)}
        self.parameter = Experiment_parameter(bounds)

        self.meta_parameter = Experiment_parameter(bounds={'beta': (15., 25.)})

        def meta_score_function(beta):

            def score_function(results):
                return results.larmor / results.t2

            def auto_characterization(epsilon=None, theta=None):
                return Results(t2=epsilon * np.sin(theta) + 0.1, larmor=(10.+(beta-20.)**2) * theta / epsilon)

            self.jean = Jean(parameters=self.parameter,
                             n_calls=10,
                             n_initial_points=5,
                             score_function=score_function,
                             measurement=auto_characterization,
                             database=Database(inner_loop_dir),
                             plot=False)
            res = self.jean()
            return Results(x=res.x, y=res.fun)

        self.meta_jean = Jean(parameters=self.meta_parameter,
                         n_calls=10,
                         n_initial_points=5,
                         score_function=lambda x: x.y,
                         measurement=meta_score_function,
                         database=Database(experiment_dir),
                         plot=True)

        self.res = self.meta_jean()
        print(self.meta_jean.measurement_results)
        print(self.meta_jean.measurement_results[tuple(self.res.x)])
        self.assertTrue(self.res)

if __name__ == '__main__':
    unittest.main()

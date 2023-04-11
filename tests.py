import unittest
from src import Database, Jean, Dummy_parameter
class MyTestCase(unittest.TestCase):
    def test_create_database(self, directory):
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
        self.jean = Jean(parameters=[self.parameter],
                         bounds=[(15.,25.)],
                         n_calls=10,
                         n_initial_points=5,
                         score_function=lambda x: 1/x.t2,
                         measurement=self.parameter.measurement,
                         database=self.database,
                         plot=True,
                         x0=[[23.295720250820295], [24.52004700612897], [19.018953133673882]],
                         y0=[1.26356045, 0.94962607, 1.05807385]
                         )
        self.assertTrue(self.jean)
class Results:
    def __init__(self, t2, larmor):
        self.t2 = t2
        self.larmor=larmor


jean = Jean(
                 parameters=[epsilon],
                 bounds=[(15.,25.)],
                 n_calls=6,
                 n_initial_points=1,
                 score_function=lambda x: 1/x.t2,
                 measurement=epsilon.measurement,
                 database=database,
                plot=True,
                x0=[[23.295720250820295], [24.52004700612897], [19.018953133673882]],
                y0=np.array([1.26356045, 0.94962607, 1.05807385])
)
res = jean()


if __name__ == '__main__':
    unittest.main()

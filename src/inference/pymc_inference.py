import matplotlib.pyplot as plt
import numpy as np

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

logger = logging.getLogger('pymc3')
logger.setLevel(logging.ERROR)

# Initialize random number generator
np.random.seed(123)

import pymc as pm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rc('image', cmap='inferno')

class Model():
    def __init__(self, **kwargs):
        if kwargs.get('priors'):
            self.priors = kwargs.get('priors')
        else:
            pass

    def construct_pymc_priors(self):
        # dtype = theano.config.floatX
        pm_model = pm.Model()
        pm_priors = []
        self.keys = []
        with pm_model:
            for key in self.priors:
                self.keys.append(key)
                dist = getattr(pm, self.priors[key]['dist'])
                pm_priors.append(dist(key, **self.priors[key]['kwargs']))#, dtype=dtype))
        return pm_model, pm_priors

class Rabi_model(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def pm_sim(self, params, x):
        t_burst = x[0]
        b = x[1]

        k_e = params[0]
        rate_T_2_rabi = params[1]
        g = params[2]
        alpha = params[3]
        beta = params[4]
        phi = params[5]

        mu_b = 9.274009994e-24
        h = 6.62607015e-34
        M = 1.e6
        f = 2.79e9

        k_e = k_e * M  # Rabi frequency at resonance
        k_e_2 = k_e ** 2  # Square of Rabi frequency at resonance
        cent_f = (g * b * mu_b) / h  # Central frequency of the transition
        delta = f - cent_f
        omega = pm.math.sqrt(delta ** 2 + k_e_2)  # generalised rabi frequency

        rate_T_2_rabi = rate_T_2_rabi * M

        p = alpha + beta * (k_e_2) * (
                pm.math.sin(phi + omega * t_burst * np.pi) * pm.math.sin(phi + omega * t_burst * np.pi)) * (
                    1 / (omega ** 2)) * pm.math.exp(
            -t_burst * rate_T_2_rabi)

        return p

class Inference():
    def __init__(self, model, return_samples, **kwargs):
        self.return_samples = return_samples
        self.model = model

        if kwargs.get('method') == 'VI':
            self.infer = self.variational_inference
            self.n_iters = kwargs.get('n_iters', 100000)
        elif kwargs.get('method') == 'MCMC':
            self.chains = kwargs.get('chains',1)
            self.burnin = kwargs.get('burnin',500)
            self.total_samples = self.burnin + self.return_samples
            self.infer = self.mcmc
        else:
            self.infer = self.variational_inference

    def likelihood(self, observations, observation_points):
        with self.pm_model:
            mu = self.model.pm_sim(self.pm_priors[:-1], observation_points)
            Y_obs = pm.Normal('Y_obs',
                              mu=mu,
                              sigma=self.pm_priors[-1],
                              observed=observations)

    def _init_inference(self, observations, observation_points):
        self.pm_model, self.pm_priors = self.model.construct_pymc_priors()
        self.likelihood(observations, observation_points)

    def mcmc(self, observations, observation_points):
        self._init_inference(observations, observation_points)
        with self.pm_model:
            self.trace = pm.sample(self.total_samples,
                                   tune=5000,
                                   chains=self.chains)

        #self.burned_trace = self.trace[-int(self.return_samples/self.chains):]

        self.posteriors = []
        for key in self.model.keys:
            self.posteriors.append(self.trace.posterior.__getattr__(key))

        return np.array(self.posteriors)

    def variational_inference(self, observations, observation_points):
        self._init_inference(observations, observation_points)
        with self.pm_model:
            self.fit = pm.fit(n=self.n_iters, method=pm.ADVI())

        self.trace = self.fit.sample(self.return_samples)

        self.posteriors = []
        for key in self.model.keys:
            self.posteriors.append(self.trace.posterior.__getattr__(key))

        return np.array(self.posteriors)


def rabi_fn(observation_points,
            k_e,
            rate_T_2_rabi,
            g,
            alpha,
            beta,
            phi,
            f = 2.79e9,
            sigma=None):
    """
    Rabi model function.

    @param observation_points:
    @param k_e:
    @param rate_T_2_rabi:
    @param g:
    @param alpha:
    @param beta:
    @param phi:
    @param sigma:
    @return:
    """
    # bohr magnetron
    mu_b = 9.274009994e-24
    # h
    h = 6.62607015e-34

    M = 1.e6

    t_burst = observation_points[..., 0]
    b = observation_points[..., 1]

    k_e = k_e * M  # Rabi frequency at resonance
    k_e_2 = k_e**2  # Square of Rabi frequency at resonance
    cent_f = (g * b * mu_b)/h # Central frequency of the transition
    delta = f - cent_f
    omega = np.sqrt(delta**2 + k_e_2)  # generalised rabi frequency

    rate_T_2_rabi = rate_T_2_rabi * M

    p = alpha + beta * (k_e_2) * (
            np.sin(phi + omega * t_burst * np.pi) * np.sin(phi + omega * t_burst * np.pi)) * (
                1 / (omega ** 2)) * np.exp(
        -t_burst * rate_T_2_rabi)

    if sigma is not None:
        return np.random.normal(p, sigma)
    else:
        return p


def plot_observations(t, b, observations, title="Observations"):
    """
    Plot the observations as a 2D image.
    @param t:
    @param b:
    @param observations:
    @param title:
    @return:
    """
    plt.imshow(observations,
               origin='lower',
               aspect='auto',
               extent=(t.min(), t.max(), b.min(), b.max()))
    plt.xlabel('t')
    plt.ylabel('b')
    plt.colorbar()
    plt.tight_layout()
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    """
    Run the inference.
    """
    method = 'mcmc'
    params = {
        "k_e": 101.,
        "rate_T_2_rabi": 65.,
        "g": 0.65,
        "alpha": 0.5,
        "beta": 0.5,
        "phi": 0.
    }
    # Drive frequency
    f = 2.79e9

    # Generate some data
    n = 100
    b = np.linspace(0.25, 0.35, n)
    t = np.linspace(0, 45e-9, n)

    points = np.array(np.meshgrid(t, b)).T.reshape(-1, 2)

    observations = rabi_fn(points, **params, sigma=0.06)#, f=f)

    plot_observations(t, b, observations.reshape([n, n]), title='Measurements')


    priors = {
        "k_e": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 50,
                "upper": 250
            }
        },
        "rate_T_2_rabi": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 0.,
                "upper": 200.
            }
        },
        "g": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 0.,
                "upper": 10.
            }
        },
        "alpha": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 0.,
                "upper": 2.
            }
        },
        "beta": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 0.,
                "upper": 2.
            }
        },
        "phi": {
            "dist": "Uniform",
            "kwargs": {
                "lower": -np.pi,
                "upper": np.pi
            }
        },
        "sigma": {
            "dist": "Uniform",
            "kwargs": {
                "lower": 0.,
                "upper": 0.1
            }
        }
    }

    method = "VI"
    n_samples = 200
    rabi_model = Rabi_model(priors=priors)
    inference = Inference(rabi_model, method=method, return_samples=n_samples)
    posteriors = inference.infer(observations, points.T)
    predicted_means = {key: item for key, item in zip(priors.keys(), np.squeeze(np.mean(posteriors, axis=-1)) )}

    predictions = rabi_fn(points, **predicted_means)#, f=f)

    plot_observations(t, b, predictions.reshape([n, n]), title='Measurements')

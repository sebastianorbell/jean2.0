import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
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

    b = observation_points[1, ...]
    t_burst = observation_points[0, ...]

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

def fit(bounds, xdata, ydata, n_samples=100):

    samples = [
        np.random.uniform(
            low=bounds[0][index],
            high=bounds[1][index],
            size=n_samples
        ) for index in range(len(bounds[0]))
    ]

    # the initial guess
    popts = []
    losses = []
    for index in tqdm(range(n_samples)):
        try:
            p0 = [item[index] for item in samples]
            res = curve_fit(
                rabi_fn,
                xdata,
                ydata,
                p0,
                bounds=bounds,
            )
            popt, pcov = res
            predictions = rabi_fn(xdata, *popt)
            loss = np.sum((predictions - ydata) ** 2)
            popts.append(popt)
            losses.append(loss)
        except:
            pass

    popt_max = popts[np.argmin(losses)]
    predictions = rabi_fn(xdata, *popt_max)
    return popt_max, predictions, losses

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

    # points = points.reshape(-1, 2)

    # Distribution for the priors
    n_samples = 500

    bounds = [
        [50, 0, 0, 0, 0, 0],
        [250, 200, 4, 2, 2, 2 * np.pi]
    ]

    popt_max, predictions, losses = fit(bounds, observations, points, n_samples=n_samples)

    plot_observations(t, b, predictions.reshape([n, n]), title='Predictions')

    # histogram of the loss
    plt.hist(losses)
    plt.show()
import numpy as np
import matplotlib.pyplot as plt


def unpack_database(database):
    return database.measurement_results, database.optimal_point, database.query_points, database.query_values, database.models
def plot_rabi_surface(measurement_results, optimal_point, query_points, query_values, models):

    t_burst = measurement_results.get(list(measurement_results.keys())[0]).t_burst
    sweep_values = np.array([float(i[0]) for i in list(measurement_results.keys())])
    signals = np.array([measurement_results[i].signal for i in list(measurement_results.keys())])
    T, F = np.meshgrid(t_burst, sweep_values)

    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios':[4, 1]})
    p0 = axes[0].pcolor(T, F, signals)#, shading='gouraud')
    axes[0].axhline(optimal_point, color="red", linestyle="dotted", label='Optimum')
    colorbar = plt.colorbar(p0, ax=axes, orientation='horizontal')
    colorbar.set_label("PCA 0 (a.u.)")
    model = models[0]
    p_copy = np.array(query_values).copy()
    p_copy[p_copy != 0.] = -1. / p_copy[p_copy != 0.]

    model.fit(np.array(query_points), p_copy)
    fs = np.linspace(np.array(query_points).min(), np.array(query_points).max(), 100)
    predictions, std = model.predict(fs.reshape(-1, 1), return_std=True)

    axes[1].scatter(p_copy, query_points, marker='*', color='red')
    axes[1].plot(predictions, fs, color='black', linestyle='--')
    axes[1].fill_betweenx(fs, predictions-(std)*2, predictions+(std)*2, color='black', alpha=0.3)
    axes[1].axhline(optimal_point, color='red', linestyle='dotted')
    # Customize the plot
    axes[0].set_xlabel('t burst')
    axes[1].set_xlabel('Rabi frequency')
    axes[0].set_ylabel('Sweep value')
    # Show the plot
    plt.show()
    r = np.cos(t_burst[np.newaxis, ...]*((1e6)*predictions[..., np.newaxis]*np.pi*2))
    plt.imshow(r)
    plt.show()

    return t_burst, sweep_values, signals

def plot_triangle(data):
    coords = [data['coords'][key]['data'] for key in data['coords'].keys()]
    vals = data['data_vars']['I_SD']['data']
    plt.imshow(vals, aspect='auto', origin='lower',
               extent=[np.min(coords[1]), np.max(coords[1]), np.min(coords[0]), np.max(coords[0])], cmap='magma')
    plt.colorbar()
    plt.show()

def plot_optimal_fit(data):
    arg = np.argwhere(optimal_results.b_field == optimal_results.resonant_field)
    f, t2, snr = fit_decaying_cosine(optimal_results.t_burst, optimal_results.pca[np.squeeze(arg)], f_min=50e6, f_max=250e6,
                        plot=True)
    return f, t2, snr

def plot_2d_rabi(pca, t_burst, b_field, scores):
    fig, axes = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [4, 1]})
    p0 = axes[0].imshow(pca, origin='lower', aspect='auto', extent=[t_burst.min(), t_burst.max(), b_field.min(), b_field.max()])

    axes[1].plot(scores, b_field, color='black', linestyle='--')

    # Customize the plot
    axes[0].set_xlabel('t burst')
    axes[1].set_xlabel('Score')
    axes[0].set_ylabel('B-field')
    plt.tight_layout()
    plt.show()

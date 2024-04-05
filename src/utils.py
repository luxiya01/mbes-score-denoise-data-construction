import numpy as np
import matplotlib.pyplot as plt

def plot_differences_between_draping_results_and_raw_points(data_path, draping_paths, labels):
    """
    Plot the differences between the draping results and the raw points.
    """
    data = np.load(data_path, allow_pickle=True)
    raw_pings = np.stack([data["X"], data["Y"], data["Z_relative"]], axis=-1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for (draping_path, label_text) in zip(draping_paths, labels):
        draping_results = np.load(draping_path)
        no_draping_res_mask = np.all(draping_results == 0, axis=-1)
        rejected_mask = data['rejected']
        combined_mask = np.logical_or(no_draping_res_mask, rejected_mask)

        differences = draping_results - raw_pings
        differences = differences[~combined_mask]

        z_differences = differences[:, -1]
        ax[0].hist(z_differences, bins=1000, alpha=.5, label=label_text)
        ax[0].set_title('Z differences')
        ax[1].hist(np.linalg.norm(differences, axis=-1), bins=1000, alpha=.5, label=label_text)
        ax[1].set_title('Norm differences')
    plt.legend()
    plt.tight_layout()
    plt.show()

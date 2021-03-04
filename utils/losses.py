'''
Set of objets that can be imported as custom losses.
Losses can be both pytorch specific and numpy specific.
'''
import numpy as np
import matplotlib.pyplot as plt

from utils import tools

def squared_error(x, x_hat, show_plot: bool=False, return_map: bool=True):
    if isinstance(x, np.ndarray):
        x_err = (x - x_hat)**2
        if show_plot:
            x_stats = tools.BatchStatistics(x)
            x_hat_stats = tools.BatchStatistics(x_hat)
            x_err_stats = tools.BatchStatistics(x_err)

            fig, ax = plt.subplots(1, 3,  figsize=(20,13))
            ax[0].imshow(tools.unstandardize_batch(x), interpolation='nearest')
            ax[0].set_title(f'x: [{x_stats.min:.2f}, {x_stats.max:.2f}]')
            ax[1].imshow(tools.unstandardize_batch(x_hat), interpolation='nearest')
            ax[1].set_title(f'x_hat: [{x_hat_stats.min:.2f}, {x_hat_stats.max:.2f}]')
            ax[2].imshow(tools.unstandardize_batch(x_err))
            ax[2].set_title(f'x_err: [{x_err_stats.min:.2f}, {x_err_stats.max:.2f}], mse: {np.mean(x_err):.2f}')
            for i in range(len(ax)): ax[i].grid(False)
            plt.show()
            del fig, ax

        if return_map:
            # Return the *mean* of the squared error map (e.g. MSE)
            # AND the error map itself
            return np.mean(x_err), x_err
        else:
            return np.mean(x_err)
    else:
         raise TypeError('Only numpy available array\'s supported')
'''
Set of objets that can be imported as custom losses.
Losses can be both pytorch specific and numpy specific.
'''
import numpy as np
import matplotlib.pyplot as plt
import torch
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

 def mse_loss(x, x_hat):
    """
    Returns the MSE between an image and its reconstruction

    INPUT:
    x: Tensor (B, C, H, W) -> source image
    x_hat: Tensor (B, C, H, W) -> reconstruction

    OUTPUT:
    mse_loss_sum: Tensor (B, 1) -> sum of the MSE for each image pair
    """
    mse_loss = torch.nn.MSELoss(reduction='none')
    recons_error = mse_loss(x, x_hat)
    mse_loss_sum = torch.sum(recons_error, dim=(1,2,3))

    return mse_loss_sum

def recons_probability(x_hat):
    """
    Returns the sum of the probabilities of each pixel in the image

    INPUT:
    x_hat: Tensor (B, C, H, W) -> reconstruction

    OUTPUT:
    recons_probability: Tensor (B, 1) -> sum of of the reconstruction
                                         probabilities in the image
    """

    probs = -torch.log(x_hat)
    sum_probs = torch.sum(probs, dim(1, 2, 3))
    return sum_probs

def kl_divergence(mu, logvar):
    """
    Returns the KLD of the sampled latent vector w.r.t. to a Standard Normal Gaussian

    INPUT:
    mu: Tensor (B, 1) -> mean of a Gaussian distribution q(z|x_test)
    logvar: Tensor (B, 1) -> logvar of a Gaussian distribution q(z|x_test)

    OUTPUT:
    kld_loss: Tensor(B, 1) -> Kullback-Leibler divergence
    """

    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

    return kld_loss

def mixed_loss(x_hat, mu, logvar):
    """
    Implements the mixed loss M1 in the Sintini, Kuntze paper

    INPUT:
    x_hat: Tensor (B, C, H, W) -> reconstruction
    mu: Tensor (B, 1) -> mean of a Gaussian distribution q(z|x_test)
    logvar: Tensor (B, 1) -> logvar of a Gaussian distribution q(z|x_test)

    OUTPUT:
    mixed_loss: Tensor
    """

    recons_prob_loss = recons_probability(x_hat)
    kld_loss = kl_divergence(mu, logvar)
    mixed_loss = recons_prob_loss + kld_loss

    return mixed_loss
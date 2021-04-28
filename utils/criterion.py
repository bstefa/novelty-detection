import torch

class ReconsError():
    """
    Takes in a batch reconstruction and original images
    and calculates the MSE between the two.

    INPUT: 
        recons: Tensor (B, C, H, W) -> batch of reconstruction images
        imges: Tensor (B, C, H, W) -> batch of original images

    OUTPUT:
        mse_loss_sum: Tensor (B, 1) -> batch of MSE between recons and images
    """
    def __init__(self):
        print("Testing with Reconstruction Error criterion")

    def __call__(self, **kwargs):

        mse_loss = torch.nn.MSELoss(reduction='none')
        recons_error = mse_loss(kwargs['recons'], kwargs['images'])
        mse_loss_sum = torch.sum(recons_error, dim=(1, 2, 3))

        return mse_loss_sum

class ReconsProbability():
    """
    Returns the sum of the probabilities of each pixel in the image

    INPUT:
        recons: Tensor (B, C, H, W) -> reconstruction

    OUTPUT:
        sum_probs: Tensor (B, 1) -> sum of of the reconstruction
                                         probabilities in the image
    """
    def __init__(self):
        print("Testing with Reconstruction Probability criterion")

    def __call__(self, **kwargs):

        # probs = torch.log(-kwargs['recons'])
        dist = torch.distributions.Normal(kwargs['recons'], 1.0)
        log_pxz = dist.log_prob(kwargs['images'])
        sum_probs = torch.sum(log_pxz, dim=(1, 2, 3))

        return -sum_probs

class KLD():
    """
    Returns the KLD of the sampled latent vector w.r.t. to a Standard Normal Gaussian

    INPUT:
    mu: Tensor (B, 1) -> mean of a Gaussian distribution q(z|x_test)
    log_var: Tensor (B, 1) -> logvar of a Gaussian distribution q(z|x_test)

    OUTPUT:
    kld: Tensor(B, 1) -> Kullback-Leibler divergence
    """
    def __init__(self):
        print("Testing with KLD criterion")

    def __call__(self, **kwargs):

        # kld for each distribution
        kld = 0.5 * torch.sum(1 + kwargs['log_var'] - kwargs['mu'] ** 2 - kwargs['log_var'].exp(), dim=1)

        # std = torch.exp(kwargs['log_var'] / 2)

        # p = torch.distributions.Normal(torch.zeros_like(kwargs['mu']), torch.ones_like(std))
        # q = torch.distributions.Normal(kwargs['mu'], std)

        # # 2. get the probabilities from the equation
        # z = q.rsample()
        # log_qzx = q.log_prob(z)
        # log_pz = p.log_prob(z)

        # # kld

        # kld = (log_qzx - log_pz)
        # kld = kld.sum(dim=(-1))

        # print(kld)

        return kld

class MixedLoss():
    """
    Implements the mixed loss M1 in the Sintini, Kuntze paper

    INPUT:
        recons: Tensor (B, C, H, W) -> reconstruction
        mu: Tensor (B, 1) -> mean of a Gaussian distribution q(z|x_test)
        log_var: Tensor (B, 1) -> logvar of a Gaussian distribution q(z|x_test)

    OUTPUT:
        mixed_loss: Tensor
    """
    def __init__(self):
        print("Testing with Mixed Loss criterion")
        self.recons_probability = ReconsProbability()
        self.kld = KLD()

    def __call__(self, **kwargs):
        
        recons_prob_loss = self.recons_probability(images=kwargs['images'], recons=kwargs['recons'])
        kld_loss = self.kld(mu=kwargs['mu'], log_var=kwargs['log_var'])

        mixed_loss = recons_prob_loss + kld_loss

        return mixed_loss
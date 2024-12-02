"""Losses."""

import torch

from neurometry.estimators.curvature.hyperspherical.distributions import (
    hyperspherical_uniform,
    von_mises_fisher,
)


def elbo(x, x_mu, posterior_params, z, labels, config):
    """Compute the ELBO for the VAE loss.

    The VAE elbo loss is defined as:
    = reconstruction loss + Kl divergence
    over all elements and batch

    Notes
    -----
    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters
    ----------
    x : array-like, shape=[batch_size, data_dim]
        Input data.
    gen_likelihood_params : tuple
        Learned distributional parameters of generative model.
        (e.g., (x_mu,x_logvar) for Gaussian).
    posterior_params : tuple
        Learned distributional parameters of approximate posterior.
        (e.g., (z_mu,z_logvar) for Gaussian).
    config : module
        Module specifying various model hyperparameters

    Returns
    -------
    _ : array-like, shape=[batch_size,]
        VAE elbo loss function on each batch element.
    """

    if config.posterior_type == "gaussian":
        z_mu, z_logvar = posterior_params
        z_var = torch.exp(z_logvar)
        kld = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_var)

    if config.posterior_type == "hyperspherical":
        z_mu, z_kappa = posterior_params
        q_z = von_mises_fisher.VonMisesFisher(z_mu, z_kappa)
        p_z = hyperspherical_uniform.HypersphericalUniform(
            config.latent_dim - 1, device=config.device
        )
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

    if config.posterior_type == "toroidal":
        z_theta_mu, z_theta_kappa, z_phi_mu, z_phi_kappa = posterior_params
        q_z_theta = von_mises_fisher.VonMisesFisher(z_theta_mu, z_theta_kappa)
        q_z_phi = von_mises_fisher.VonMisesFisher(z_phi_mu, z_phi_kappa)
        p_z = hyperspherical_uniform.HypersphericalUniform(
            config.latent_dim - 1, device=config.device
        )
        kld_theta = torch.distributions.kl.kl_divergence(q_z_theta, p_z).mean()
        kld_phi = torch.distributions.kl.kl_divergence(q_z_phi, p_z).mean()
        kld = kld_theta + kld_phi

    if config.gen_likelihood_type == "gaussian":
        recon_loss = torch.mean((x - x_mu).pow(2))
    else:
        raise NotImplementedError

    if config.dataset_name == "s1_synthetic":
        recon_loss = recon_loss / (config.radius**2)

    latent_loss = latent_regularization_loss(labels, z, config)
    moving_loss = moving_forward_loss(z, config)

    elbo_loss = (
        config.alpha * recon_loss
        + config.beta * kld
        + config.gamma * latent_loss
        + config.gamma_moving * moving_loss
    )
    return elbo_loss, recon_loss, kld, latent_loss, moving_loss


def latent_regularization_loss(labels, z, config):
    """Compute squared geodesic distance between outside and inside's variables.

    For example, this computes the squared difference in angles between the lab's
    angle and the latent angle.

    Parameters
    ----------
    labels : array-like, shape=[batch_size, latent_dim]
        Task variables recorded.
    z : array-like, shape=[batch_size, latent_dim]
        Latent variables on the template manifold.
    config : object-like
        Configuration of the experiment in wandb format.

    Returns
    -------
    _ : array-like, shape=[batch_size, 1]
        Squared geodesic distance, i.e. the loss.
    """
    if config.dataset_name == "s1_synthetic":
        latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
        latent_loss = angle_loss
    elif config.dataset_name in ("experimental", "three_place_cells_synthetic"):
        labels = labels * (torch.pi / 180)
        latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
        latent_loss = angle_loss
    elif config.dataset_name == "s2_synthetic":
        latent_thetas = torch.arccos(z[:, 2])
        latent_phis = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        thetas_loss = torch.mean(1 - torch.cos(latent_thetas - labels[:, 0]))
        phis_loss = torch.mean(
            torch.sin(latent_thetas)
            * torch.sin(labels[:, 0])
            * (1 - torch.cos(latent_phis - labels[:, 1]))
        )
        latent_loss = thetas_loss + phis_loss
    elif config.dataset_name == "t2_synthetic":
        latent_thetas = (
            torch.atan2(z[:, 2], 2 - torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2))
            + 2 * torch.pi
        ) % (2 * torch.pi)
        latent_phis = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
        thetas_loss = torch.mean(1 - torch.cos(latent_thetas - labels[:, 0]))
        phis_loss = torch.mean(1 - torch.cos(latent_phis - labels[:, 1]))
        latent_loss = thetas_loss + phis_loss
    elif config.dataset_name == "grid_cells":
        return 0

    return latent_loss**2


def moving_forward_loss(z, config):
    """Compute difference between two consecutive z's.

    This loss will enforce that the latent variable (the angles) only increase
    when the rat moves forward.

    In order to enforce increasing values of z's, the loss is not squared.
    minimizing -(z_t+1 - z_t) will force it to be negative, i.e. z_t+1 > z_t.

    We remove the situation where the rat crosses the angles 360 --> 0.
    Note that atol=0.089 radians corresponds to 5 degrees, which is the max degree
    difference observed in the rat's labelled lab angles.

    Parameters
    ----------
    """
    if config.dataset_name != "experimental":
        # print(
        #     "WARNING: Moving forward loss only implemented for experimental data
        #     --> Skipped."
        # )
        return torch.zeros(1).to(config.device)
    if len(z) == 1:
        return torch.zeros(1).to(config.device)
    latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (2 * torch.pi)
    diff = latent_angles[1:] - latent_angles[:-1]
    # only keep angles where the rat is not crossing 360 --> 0
    mask = ~torch.isclose(
        2 * torch.pi - latent_angles[:-1], torch.tensor(0.0), atol=0.089
    )
    loss = -diff[mask]
    if len(loss) == 0:
        return torch.zeros(1).to(config.device)
    return torch.mean(loss)

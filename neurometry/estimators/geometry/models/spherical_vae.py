"""G-manifold deep learning.

This file gathers deep learning models related to G-manifold learning.
"""

import torch
from torch.nn import functional as F

from neurometry.estimators.geometry.models import train_config as train_config
from neurometry.estimators.geometry.models.hyperspherical.distributions import (
    hyperspherical_uniform,
    von_mises_fisher,
)


class SphericalVAE(torch.nn.Module):
    """VAE with Linear (fully connected) layers.

    Parameters
    ----------
    extrinsic_dim : int
        Extrinsic dimension of neural data.
        This is the dimension of a hyperplane in which neural activity lies within neural state space.
    latent_dim : int
        Dimension of the latent space. This is the dimension of the minimal euclidean embedding of the task-relevant variables.
    """

    def __init__(
        self,
        extrinsic_dim,
        latent_dim,
        sftbeta=4.5,
        encoder_width=400,
        encoder_depth=2,
        decoder_width=400,
        decoder_depth=2,
        drop_out_p=0.0,
    ):
        super().__init__()
        self.extrinsic_dim = extrinsic_dim
        self.sftbeta = sftbeta
        self.latent_dim = latent_dim
        self.drop_out_p = drop_out_p

        decoder_width = encoder_width
        decoder_depth = encoder_depth

        self.encoder_fc = torch.nn.Linear(self.extrinsic_dim, encoder_width)
        self.encoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(encoder_width, encoder_width)
                for _ in range(encoder_depth)
            ]
        )

        self.fc_z_mu = torch.nn.Linear(encoder_width, self.latent_dim)
        self.fc_z_logvar = torch.nn.Linear(encoder_width, 1)  # kappa

        self.decoder_fc = torch.nn.Linear(self.latent_dim, decoder_width)
        self.decoder_linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(decoder_width, decoder_width)
                for _ in range(decoder_depth)
            ]
        )

        self.fc_x_mu = torch.nn.Linear(decoder_width, self.extrinsic_dim)

        self.drop_out = torch.nn.Dropout(p=self.drop_out_p)

    def encode(self, x):
        """Encode input into mean and log-variance.

        The parameters mean (mu) and variance (computed
        from logvar) defines a multivariate Gaussian
        that represents the approximate posterior of the
        latent variable z given the input x.

        Parameters
        ----------
        x : array-like, shape=[batch_size, extrinsic_dim]
            Input data.

        Returns
        -------
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """
        h = F.softplus(self.encoder_fc(x.double()), beta=self.sftbeta)

        for _i_layer, layer in enumerate(self.encoder_linears):
            # h = self.drop_out(h)
            h = layer(h)
            h = F.softplus(h, beta=self.sftbeta)

        z_mu = self.fc_z_mu(h)
        z_kappa = F.softplus(self.fc_z_logvar(h)) + 1

        return z_mu, z_kappa

    def reparameterize(self, posterior_params):
        """
        Apply reparameterization trick. We 'externalize' the
        randomness in z by re-parameterizing the variable as
        a deterministic and differentiable function of x,
        the encoder weights, and a new random variable eps.

        Parameters
        ----------
        posterior_params : tuple
            Distributional parameters of approximate posterior. ((e.g.), (z_mu,z_logvar) for Gaussian.

        Returns
        -------

        z: array-like, shape = [batch_size, latent_dim]
            Re-parameterized latent variable.
        """

        z_mu, z_kappa = posterior_params
        q_z = von_mises_fisher.VonMisesFisher(z_mu, z_kappa)

        return q_z.rsample()

    def decode(self, z):
        """Decode latent variable z into data.

        Parameters
        ----------
        z : array-like, shape=[batch_size, latent_dim]
            Input to the decoder.

        Returns
        -------
        _ : array-like, shape=[batch_size, extrinsic_dim]
            Reconstructed data corresponding to z.
        """

        h = F.softplus(self.decoder_fc(z), beta=self.sftbeta)

        for _i_layer, layer in enumerate(self.decoder_linears):
            # h = self.drop_out(h)
            h = layer(h)
            h = F.softplus(h, beta=self.sftbeta)

        return self.fc_x_mu(h)

    def forward(self, x):
        """Run VAE: Encode, sample and decode.

        Parameters
        ----------
        x : array-like, shape=[batch_size, extrinsic_dim]
            Input data.

        Returns
        -------
        _ : array-like, shape=[batch_size, extrinsic_dim]
            Reconstructed data corresponding to z.
        mu : array-like, shape=[batch_size, latent_dim]
            Mean of multivariate Gaussian in latent space.
        logvar : array-like, shape=[batch_size, latent_dim]
            Vector representing the diagonal covariance of the
            multivariate Gaussian in latent space.
        """

        posterior_params = self.encode(x)
        z = self.reparameterize(posterior_params)
        x_mu = self.decode(z)
        return z, x_mu, posterior_params

    def _elbo(self, x, x_mu, posterior_params):
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

        Returns
        -------
        _ : array-like, shape=[batch_size,]
            VAE elbo loss function on each batch element.
        """

        z_mu, z_kappa = posterior_params
        q_z = von_mises_fisher.VonMisesFisher(z_mu, z_kappa)
        p_z = hyperspherical_uniform.HypersphericalUniform(
            self.latent_dim - 1, device=train_config.device
        )
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()

        recon_loss = torch.mean((x - x_mu).pow(2))

        return train_config.recon_weight * recon_loss + train_config.kld_weight * kld

    def _latent_regularization_loss(self, labels, z):
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
        if self.latent_dim == 2:
            latent_angles = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (
                2 * torch.pi
            )
            angle_loss = torch.mean(1 - torch.cos(latent_angles - labels))
            latent_loss = angle_loss
        elif self.latent_dim == 3:
            latent_thetas = torch.arccos(z[:, 2])
            latent_phis = (torch.atan2(z[:, 1], z[:, 0]) + 2 * torch.pi) % (
                2 * torch.pi
            )
            thetas_loss = torch.mean(1 - torch.cos(latent_thetas - labels[:, 0]))
            phis_loss = torch.mean(
                torch.sin(latent_thetas)
                * torch.sin(labels[:, 0])
                * (1 - torch.cos(latent_phis - labels[:, 1]))
            )
            latent_loss = thetas_loss + phis_loss

        return latent_loss**2

    def criterion(self, x, x_mu, posterior_params, labels, z):
        elbo_loss = self._elbo(x, x_mu, posterior_params)
        latent_loss = self._latent_regularization_loss(labels, z)

        return elbo_loss + train_config.latent_weight * latent_loss

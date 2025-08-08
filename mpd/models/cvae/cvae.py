# https://github.com/chendaichao/VAE-pytorch/blob/master/Models/VAE/model.py
from copy import copy
from math import prod

import einops
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict

from einops.layers.torch import Rearrange

from mpd.models import PreNorm, Downsample1d, Conv1dBlock, group_norm_n_groups, Upsample1d, MLP, ResidualTemporalBlock
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class Encoder(nn.Module):
    def __init__(
        self,
        n_support_points=None,
        state_dim=None,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        conditioning_embed_dim=4,
        **kwargs,
    ):
        super().__init__()

        self.state_dim = state_dim
        input_dim = state_dim

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        # conditioning dimension
        cond_dim = conditioning_embed_dim

        # Unet downsampling
        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, cond_dim, None),
                        ResidualTemporalBlock(dim_out, dim_out, cond_dim, None),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                n_support_points = n_support_points // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, None)
        self.out_shape = (mid_dim, n_support_points)

    def forward(self, x, context):
        """
        x : [ batch x horizon x state_dim ]
        context: [batch x context_dim]
        """
        b, h, d = x.shape

        c_emb = context

        # swap horizon and channels (state_dim)
        x = einops.rearrange(x, "... h c -> ... c h")  # batch, horizon, channels (state_dim)

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, c_emb)
            x = resnet2(x, c_emb)
            x = downsample(x)

        x = self.mid_block1(x, c_emb)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_support_points=None,
        state_dim=None,
        unet_input_dim=32,
        dim_mults=(1, 2, 4, 8),
        conditioning_embed_dim=4,
        latent_dim=64,
        in_shape=(128, 2),
        **kwargs,
    ):
        super().__init__()

        self.in_shape = in_shape

        self.state_dim = state_dim
        input_dim = state_dim

        dims = [input_dim, *map(lambda m: unet_input_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        # conditioning dimension (time + context)
        cond_dim = conditioning_embed_dim

        # Unet upsampling
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        mid_dim = dims[-1]
        self.latent_to_mid = MLP(latent_dim, prod(self.in_shape), hidden_dim=mid_dim, n_layers=1, act="relu")
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim, None)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out, dim_in, cond_dim, None),
                        ResidualTemporalBlock(dim_in, dim_in, cond_dim, None),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )
            if not is_last:
                n_support_points = n_support_points * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(unet_input_dim, unet_input_dim, kernel_size=5, n_groups=group_norm_n_groups(unet_input_dim)),
            nn.Conv1d(unet_input_dim, state_dim, 1),
        )

    def forward(self, z, context):
        z = self.latent_to_mid(z)

        z = einops.rearrange(z, "... (c h) -> ... c h", c=self.in_shape[0], h=self.in_shape[1])
        c_emb = context

        x = self.mid_block2(z, c_emb)

        for resnet, resnet2, upsample in self.ups:
            x = resnet(x, c_emb)
            x = resnet2(x, c_emb)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "... c h -> ... h c")
        return x


class CVAEModel(nn.Module):
    def __init__(
        self,
        context_model,
        cvae_latent_dim=64,
        logvar_min=-7,
        logvar_max=2,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        super(CVAEModel, self).__init__()

        self.context_model = context_model

        self.encoder = Encoder(**kwargs)
        self.decoder = Decoder(latent_dim=cvae_latent_dim, in_shape=self.encoder.out_shape, **kwargs)

        self.latent_dim = cvae_latent_dim
        self.latent_net = nn.Linear(prod(self.encoder.out_shape), 2 * cvae_latent_dim)

        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.softplus = nn.Softplus()

        self.tensor_args = tensor_args

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape, **self.tensor_args)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x, context_d, **kwargs):
        c_emb = self.context_model(**context_d)
        z_encoder = self.encoder(x, c_emb)

        guassian_params = self.latent_net(einops.rearrange(z_encoder, "... h c -> ... (h c)"))
        gaussian_mean, gaussian_logvar = guassian_params.chunk(2, dim=-1)

        # clip log variance in a differentiable way - https://arxiv.org/pdf/2109.14311.pdf (4.2)
        gaussian_logvar = self.logvar_min + self.softplus(
            self.logvar_max - self.softplus(self.logvar_max - gaussian_logvar) - self.logvar_min
        )

        z_sample = self.sampling(gaussian_mean, gaussian_logvar)

        x_pred = self.decoder(z_sample, c_emb)

        return x_pred, gaussian_mean, gaussian_logvar

    def loss(self, x, input_dict, loss_cvae_kl_weight=1e-1, **kwargs):
        x_pred, guassian_mean, guassian_logvar = self.forward(x, input_dict)

        reconstruction_loss = self.reconstruction_loss_fn(x_pred, x)
        kl_divergence = 0.5 * torch.sum(-1 - guassian_logvar + torch.exp(guassian_logvar) + guassian_mean**2, dim=-1)
        kl_divergence_weighted = loss_cvae_kl_weight * kl_divergence

        loss = (reconstruction_loss + kl_divergence_weighted).mean()
        info = {
            "reconstruction_loss": reconstruction_loss.mean(),
            "kl_divergence": kl_divergence_weighted.mean(),
        }
        return loss, info

    @staticmethod
    def reconstruction_loss_fn(x_pred, x):
        assert x_pred.ndim >= 3, f"x_pred.ndim={x_pred.ndim}"  # [batch x horizon x state_dim]
        return torch.linalg.vector_norm(x_pred - x, dim=(-1, -2))

    def run_inference(
        self, context_d=None, hard_conds=None, n_samples=1, return_chain=False, return_chain_x_recon=False, **kwargs
    ):
        # repeat contexts for n_samples
        for k, v in context_d.items():
            context_d[k] = einops.repeat(v, "... -> b ...", b=n_samples)
        c_emb = self.context_model(**context_d)
        z_sample = torch.randn((n_samples, self.latent_dim), **self.tensor_args)
        x_sample = self.decoder(z_sample, c_emb)
        x_sample = x_sample.contiguous()
        return x_sample[None, ...]  # [1 x n_samples x horizon x state_dim], dummy dimension to match diffusion step

    # ------------------------------------------ warmup ------------------------------------------#
    @torch.no_grad()
    def warmup(self, shape_x, device="cuda"):
        batch_size, n_support_points, state_dim = shape_x
        c_emb = None
        if self.context_model is not None:
            c_emb = torch.randn(batch_size, self.context_model.out_dim, device=device)
        z_sample = torch.randn((batch_size, self.latent_dim), **self.tensor_args)
        _ = self.decoder(z_sample, c_emb)

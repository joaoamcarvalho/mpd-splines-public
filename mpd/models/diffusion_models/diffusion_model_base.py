"""
Adapted from https://github.com/jannerm/diffuser
"""

import abc
import time
from collections import namedtuple
from copy import copy
from functools import partial

import einops
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from abc import ABC

from torch.nn import DataParallel

from mpd.models.diffusion_models.helpers import cosine_beta_schedule, Losses, exponential_beta_schedule
from mpd.models.diffusion_models.sample_functions import (
    extract,
    guide_gradient_steps,
    ddpm_sample_fn,
    apply_hard_conditioning,
    ddim_create_time_pairs,
)
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, clip_grad_by_norm, clip_grad_by_value


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class GaussianDiffusionModel(nn.Module, ABC):

    def __init__(
        self,
        denoise_fn=None,
        variance_schedule="cosine",
        n_diffusion_steps=100,
        clip_denoised=True,
        predict_epsilon=False,
        loss_type="l2",
        context_model=None,
        **kwargs,
    ):
        super().__init__()

        self.model = MyDataParallel(denoise_fn)

        self.context_model = context_model

        self.n_diffusion_steps = n_diffusion_steps

        self.state_dim = self.model.state_dim

        if variance_schedule == "cosine":
            betas = cosine_beta_schedule(n_diffusion_steps, s=0.008, a_min=0, a_max=0.999)
        elif variance_schedule == "exponential":
            betas = exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0)
        else:
            raise NotImplementedError

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        ## get loss coefficients and initialize objective
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    def predict_noise_from_start(self, x_t, t, x0):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return x0
        else:
            return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
                self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
            )

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def predict_x_recon(self, x_t, t, context_d):
        context_emb = None
        if self.context_model is not None:
            context_emb = self.context_model(**context_d)

        x_recon = self.predict_start_from_noise(x_t, t=t, noise=self.model(x_t, t, context_emb))

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()
        return x_recon

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, hard_conds, context_d, t, prior_weight_with_guide=1.0, **kwargs):
        context_emb = None
        if self.context_model is not None:
            context_emb = self.context_model(**context_d)

        noise_pred = self.model(x, t, context_emb)
        noise_pred = prior_weight_with_guide * noise_pred  # weight the prior noise
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if self.clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape_x,
        hard_conds,
        context_d=None,
        return_chain=False,
        return_chain_x_recon=False,
        sample_fn=ddpm_sample_fn,
        n_diffusion_steps_without_noise=0,
        t_start_guide=torch.inf,
        **sample_kwargs,
    ):
        device = self.betas.device

        batch_size = shape_x[0]
        x = torch.randn(shape_x, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None
        chain_x_recon = [] if return_chain_x_recon else None

        for i in reversed(range(-n_diffusion_steps_without_noise, self.n_diffusion_steps)):
            t = make_timesteps(batch_size, i, device)
            x, x_recon = sample_fn(
                self, x, hard_conds, context_d, t, activate_guide=True if i <= t_start_guide else False, **sample_kwargs
            )
            x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x)
            if return_chain_x_recon:
                chain_x_recon.append(x_recon)

        chains = []
        if return_chain:
            chain = torch.stack(chain, dim=1)
            chains.append(chain)

        if return_chain_x_recon:
            chain_x_recon = torch.stack(chain_x_recon, dim=1)
            chains.append(chain_x_recon)

        return x, *chains

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        shape_x,
        hard_conds,
        context_d=None,
        return_chain=False,
        return_chain_x_recon=False,
        ddim_eta=0.0,
        ddim_skip_type="uniform",
        ddim_sampling_timesteps=None,
        t_start_guide=torch.inf,
        scale_grad_by_one_minus_alpha=False,
        guide=None,
        guide_lr=0.05,
        n_guide_steps=1,
        max_perturb_x=0.1,
        clip_grad=False,
        clip_grad_rule="value",  # 'norm', 'value'
        max_grad_norm=1.0,  # clip the norm of the control point gradients
        max_grad_value=1.0,  # clip the control point gradients
        n_diffusion_steps_without_noise=0,
        ddim_scale_grad_prior=1.0,
        compute_costs_with_xrecon=False,
        results_ns=None,
        **sample_kwargs,
    ):
        # Adapted from https://github.com/ezhang7423/language-control-diffusion/blob/63cdafb63d166221549968c662562753f6ac5394/src/lcd/models/diffusion.py#L226
        context_emb = None
        if context_d is not None:
            context_emb = self.context_model(**context_d)

        device = self.betas.device
        batch_size = shape_x[0]
        total_timesteps = self.n_diffusion_steps
        sampling_timesteps = ddim_sampling_timesteps if ddim_sampling_timesteps is not None else total_timesteps
        assert (
            sampling_timesteps <= total_timesteps
        ), f"sampling_timesteps={sampling_timesteps} > total_timesteps={total_timesteps}"

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        time_pairs = ddim_create_time_pairs(
            total_timesteps, sampling_timesteps, ddim_skip_type, n_diffusion_steps_without_noise
        )

        clip_grad_fn = lambda x: x
        if clip_grad and clip_grad_rule == "norm":
            clip_grad_fn = partial(clip_grad_by_norm, max_grad_norm=max_grad_norm)
        elif clip_grad and clip_grad_rule == "value":
            clip_grad_fn = partial(clip_grad_by_value, max_grad_value=max_grad_value)

        x = torch.randn(shape_x, device=device)
        x = apply_hard_conditioning(x, hard_conds)

        chain = [x] if return_chain else None
        chain_x_recon = [x] if return_chain_x_recon else None

        for k_step, (_time, _time_next) in enumerate(time_pairs):
            if _time == _time_next:
                continue
            if _time_next < 0:
                _time = 1
                _time_next = 0

            t = make_timesteps(batch_size, _time, device)
            t_next = make_timesteps(batch_size, _time_next, device)

            alpha = extract(self.alphas_cumprod, t, x.shape)
            alpha_next = extract(self.alphas_cumprod, t_next, x.shape)
            sigma = ddim_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            # denoising noise
            with TimerCUDA() as t_generator:
                model_out = self.model(x, t, context_emb)

            if results_ns is not None:
                results_ns.t_generator += t_generator.elapsed

            grad_prior = model_out

            def update_x(_x, _grad_prior):
                _x_recon = self.predict_start_from_noise(_x, t=t, noise=_grad_prior)
                if self.clip_denoised:
                    _x_recon.clamp_(-1.0, 1.0)
                else:
                    assert RuntimeError()
                _pred_noise = self.predict_noise_from_start(_x, t=t, x0=_grad_prior)
                _x = _x_recon * alpha_next.sqrt() + c * _pred_noise
                _x = apply_hard_conditioning(_x, hard_conds)
                return _x, _x_recon

            # Modify the noise if guidance is active
            # https://arxiv.org/pdf/2105.05233.pdf - Algorithm 2
            if guide is not None and (ddim_sampling_timesteps - k_step) <= t_start_guide:
                with TimerCUDA() as t_guide:
                    x_start = x.clone()
                    for k_gd in range(n_guide_steps):
                        grad_prior_weighted = grad_prior * ddim_scale_grad_prior
                        if compute_costs_with_xrecon:
                            raise NotImplementedError("compute_costs_with_xrecon is not implemented")
                        else:
                            grad_guide = guide(x, context_d=context_d)

                        grad_guide_clipped = clip_grad_fn(grad_guide)
                        grad_guide_clipped_weighted = guide_lr * grad_guide_clipped

                        # grad_prior_weighted_norm = torch.linalg.norm(grad_prior_weighted, dim=-1)
                        # grad_guide_clipped_weighted_norm = torch.linalg.norm(grad_guide_clipped_weighted, dim=-1)
                        # print(f'denoising epsilon (norm): {grad_prior_weighted_norm.mean():.4f} +- {grad_prior_weighted_norm.std():.4f}')
                        # print(f'guide grad (norm): {grad_guide_clipped_weighted_norm.mean():.4f} +- {grad_guide_clipped_weighted_norm.std():.4f}')

                        if scale_grad_by_one_minus_alpha:
                            # by default we skip it, because (1-alpha) -> 0 when t -> 0
                            grad_total = grad_prior_weighted - (1 - alpha).sqrt() * grad_guide_clipped_weighted
                        else:
                            grad_total = grad_prior_weighted - grad_guide_clipped_weighted

                        x_tmp, x_recon = update_x(x, grad_total)
                        # Clip the perturbation to avoid large changes from x_start_opt
                        x_delta = x_tmp - x_start
                        x_delta_clipped = torch.clip(x_delta, -max_perturb_x, max_perturb_x)
                        x = x_start + x_delta_clipped

                    if results_ns is not None:
                        results_ns.t_guide += t_guide.elapsed
            else:
                x, x_recon = update_x(x, grad_prior)

            if _time_next >= 1:
                # add noise
                noise = torch.randn_like(x) if ddim_eta != 0.0 else 0.0
                x = x + sigma * noise
                x = apply_hard_conditioning(x, hard_conds)

            if return_chain:
                chain.append(x.clone())
            if return_chain_x_recon:
                chain_x_recon.append(x_recon.clone())

        chains = []
        if return_chain:
            chain = torch.stack(chain, dim=1)
            chains.append(chain)

        if return_chain_x_recon:
            chain_x_recon = torch.stack(chain_x_recon, dim=1)
            chains.append(chain_x_recon)

        return x, *chains

    @torch.no_grad()
    def conditional_sample(self, hard_conds, horizon=None, batch_size=1, method="ddpm", **sample_kwargs):
        """
        hard conditions : hard_conds : { (time, state), ... }
        """
        horizon = horizon or self.horizon
        shape_x = (batch_size, horizon, self.state_dim)

        if method == "ddim":
            assert self.predict_epsilon, "ddim only works with predict_epsilon=True, because of guidance"
            return self.ddim_sample_loop(shape_x, hard_conds, **sample_kwargs)
        elif method == "ddpm":
            return self.p_sample_loop(shape_x, hard_conds, sample_fn=ddpm_sample_fn, **sample_kwargs)
        else:
            raise NotImplementedError

    def forward(self, cond, *args, **kwargs):
        raise NotImplementedError
        return self.conditional_sample(cond, *args, **kwargs)

    @torch.no_grad()
    def run_inference(
        self,
        context_d=None,
        hard_conds=None,
        n_samples=1,
        return_chain=False,
        return_chain_x_recon=False,
        **diffusion_kwargs,
    ):
        # repeat hard conditions and contexts for n_samples
        for k, v in hard_conds.items():
            new_state = einops.repeat(v, "... -> b ...", b=n_samples)
            hard_conds[k] = new_state

        for k, v in context_d.items():
            context_d[k] = einops.repeat(v, "... -> b ...", b=n_samples)

        # Sample from diffusion model
        samples, chain, chain_x_recon = self.conditional_sample(
            hard_conds,
            context_d=context_d,
            batch_size=n_samples,
            return_chain=True,
            return_chain_x_recon=True,
            **diffusion_kwargs,
        )

        # chain: [ n_samples x (n_diffusion_steps + 1) x horizon x (state_dim)]
        # extract normalized trajectories
        trajs_chain_normalized = chain
        trajs_x_recon_chain_normalized = chain_x_recon

        # trajs: [ (n_diffusion_steps + 1) x n_samples x horizon x state_dim ]
        trajs_chain_normalized = einops.rearrange(trajs_chain_normalized, "b diffsteps ... -> diffsteps b ...")
        trajs_x_recon_chain_normalized = einops.rearrange(
            trajs_x_recon_chain_normalized, "b diffsteps ... -> diffsteps b ..."
        )

        if return_chain and return_chain_x_recon:
            return trajs_chain_normalized, trajs_x_recon_chain_normalized
        elif return_chain:
            return trajs_chain_normalized

        # return the last denoising step
        return trajs_chain_normalized[-1]

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, context_d, t, hard_conds):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_hard_conditioning(x_noisy, hard_conds)

        # context model
        context_emb = None
        if self.context_model is not None:
            context_emb = self.context_model(**context_d)

        # diffusion model
        x_recon = self.model(x_noisy, t, context_emb)
        x_recon = apply_hard_conditioning(x_recon, hard_conds)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, context_d, *args):
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=x.device).long()
        return self.p_losses(x, context_d, t, *args)

    # ------------------------------------------ warmup ------------------------------------------#
    @torch.no_grad()
    def warmup(self, shape_x, device="cuda"):
        batch_size, n_support_points, state_dim = shape_x
        x = torch.randn(shape_x, device=device)
        t = make_timesteps(batch_size, 1, device)
        context_emb = None
        if self.context_model is not None:
            context_emb = torch.randn(batch_size, self.context_model.out_dim, device=device)
        self.model(x, t, context=context_emb)

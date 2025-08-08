from functools import partial

import numpy as np
import torch

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import clip_grad_by_value, clip_grad_by_norm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def apply_hard_conditioning(x, conditions):
    for k, v in conditions.items():
        x[:, k, :] = v.clone()
    return x


@torch.no_grad()
def ddpm_sample_fn(
    model,
    x,
    hard_conds,
    context_d,
    t,
    guide=None,
    n_guide_steps=1,
    scale_grad_by_std=False,
    activate_guide=False,
    prior_weight_with_guide=1.0,
    noise_std=1.0,
    noise_std_extra_schedule_fn=None,
    compute_costs_with_xrecon=False,
    results_ns=None,
    debug=False,
    **kwargs,
):
    with TimerCUDA() as t_generator:
        t_single = t[0]
        if t_single < 0:
            t = torch.zeros_like(t)

        model_mean, _, model_log_variance, x_recon = model.p_mean_variance(
            x=x,
            hard_conds=hard_conds,
            context_d=context_d,
            t=t,
            prior_weight_with_guide=prior_weight_with_guide if activate_guide else 1.0,
        )

        x = model_mean
        model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)

    if results_ns is not None:
        results_ns.t_generator += t_generator.elapsed

    with TimerCUDA() as t_guide:
        if guide is not None and activate_guide:
            x = guide_gradient_steps(
                x,
                t=t,
                model=model,
                hard_conds=hard_conds,
                context_d=context_d,
                guide=guide,
                n_guide_steps=n_guide_steps,
                scale_grad_by_std=scale_grad_by_std,
                model_var=model_var,
                compute_costs_with_xrecon=compute_costs_with_xrecon,
                debug=False,
                **kwargs,
            )
    if results_ns is not None:
        results_ns.t_guide += t_guide.elapsed

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    # For smoother results, we can decay the noise standard deviation throughout the diffusion
    # this is equivalent to using a temperature schedule in the prior distribution
    if noise_std_extra_schedule_fn is not None:
        noise_std = noise_std_extra_schedule_fn(t_single)

    return x + model_std * noise * noise_std, x_recon


@torch.enable_grad()
def guide_gradient_steps(
    x,
    t=None,
    model=None,
    hard_conds=None,
    context_d=None,
    guide=None,
    guide_lr=0.05,
    n_guide_steps=1,
    max_perturb_x=0.1,
    clip_grad=False,
    clip_grad_rule="value",  # 'norm', 'value'
    max_grad_norm=1.0,  # clip the norm of the control point gradients
    max_grad_value=1.0,  # clip the control point gradients
    scale_grad_by_std=False,
    model_var=None,
    compute_costs_with_xrecon=False,
    return_chain_x=False,
    debug=False,
    **kwargs,
):
    chain = []

    x_start = x.clone()
    x_opt = x.clone()
    x_opt.requires_grad_(True)
    opt = torch.optim.SGD([x_opt], lr=guide_lr)

    clip_grad_fn = lambda x: x
    if clip_grad and clip_grad_rule == "norm":
        clip_grad_fn = partial(clip_grad_by_norm, max_grad_norm=max_grad_norm)
    elif clip_grad and clip_grad_rule == "value":
        clip_grad_fn = partial(clip_grad_by_value, max_grad_value=max_grad_value)

    for _ in range(n_guide_steps):
        if compute_costs_with_xrecon:
            # https://arxiv.org/pdf/2407.00451 -- equation 2
            raise NotImplementedError("compute_costs_with_xrecon is not implemented")
            with torch.enable_grad():
                x_opt.requires_grad_(True)
                x_recon = model.predict_x_recon(x_opt, t, context_d)
                grad_x_recon_wrt_x = torch.autograd.grad(x_recon.sum(), x_opt)[0]
                g = guide(x_recon, context_d=context_d) * grad_x_recon_wrt_x
                g = clip_grad_by_value(g, max_grad_value=0.1)
                grad_guide = weight * g
        else:
            grad_guide = guide(x_opt, context_d=context_d)

        if scale_grad_by_std:
            grad_guide = model_var * grad_guide

        grad_guide_clipped = clip_grad_fn(grad_guide)

        # manually set the gradient and update x
        # -1 because we want to maximize the guide
        x_opt.grad = -1.0 * grad_guide_clipped
        opt.step()
        opt.zero_grad()
        x_opt.grad = None

        # Clip the perturbation to avoid large changes from x_start
        x_delta = x_opt - x_start
        x_delta_clipped = torch.clip(x_delta, -max_perturb_x, max_perturb_x)
        x_opt = x_start + x_delta_clipped

        x_opt = apply_hard_conditioning(x_opt, hard_conds)
        chain.append(x_opt) if return_chain_x else None

    if return_chain_x:
        return chain

    return x_opt


def ddim_create_time_pairs(
    total_timesteps, sampling_timesteps, ddim_skip_type="uniform", sampling_timesteps_without_noise=0
):
    # https://github.com/ermongroup/ddim/blob/8fd2b0ded231d2bcf7b4f1e296e9f946e72b7537/runners/diffusion.py#L343
    if ddim_skip_type == "uniform":
        seq = np.linspace(0, total_timesteps - 1, num=sampling_timesteps)
    elif ddim_skip_type == "quadratic":
        seq = np.linspace(0, np.sqrt(total_timesteps * 0.9), sampling_timesteps) ** 2
    elif ddim_skip_type == "exponential":
        seq = np.linspace(0, np.log(total_timesteps * 0.9), sampling_timesteps)
        seq = np.exp(seq)
        seq[0] = 0
    else:
        raise ValueError(f"Unknown ddim_skip_type: {ddim_skip_type}")

    times = seq.astype(int)

    # remove duplicates, add non duplicates randomly, and sort
    times_unique = np.unique(times)
    if times_unique.size == sampling_timesteps:
        times = times_unique
    else:
        # add random non unique times
        times_all = np.arange(total_timesteps)
        times_not_in_unique = np.where(~np.isin(times_all, times_unique))[0]
        times_not_in_unique_sample = np.random.choice(
            times_not_in_unique, sampling_timesteps - times_unique.size, replace=False
        )
        # merge and sort unique and non unique sample
        times = np.sort(np.concatenate((times_unique, times_not_in_unique_sample)))

    # Add the time pair (0, -1) to guarantee that the last step is computed with t=0
    times = np.insert(times, 0, -1)
    times = list(reversed(times.tolist()))

    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    # Add timesteps to sample the diffusion at t=0 multiple times without adding noise
    time_pairs.extend((0, -1) for _ in range(sampling_timesteps_without_noise))

    return time_pairs

import einops
from scipy import ndimage

import isaacgym
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from torch_robotics.environments import EnvPlanar2Link
from torch_robotics.robots import RobotPlanar2Link
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, get_torch_device, to_numpy

from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def plot_configuration_free_space(
    task,
    q_random=None,
    q_dim0=0,
    q_dim1=1,
    fig=None,
    ax=None,
    N_meshgrid=600,
    seed=42,
    smooth_sigma=0.0,
    use_imshow=True,
    **kwargs,
):
    fix_random_seed(seed)

    robot = task.robot

    q_limits_low = robot.q_pos_min
    q_limits_high = robot.q_pos_max
    q_limits_low_np = to_numpy(q_limits_low)
    q_limits_high_np = to_numpy(q_limits_high)

    # create meshgrid
    q1 = torch.linspace(q_limits_low[q_dim0], q_limits_high[q_dim0], N_meshgrid, **robot.tensor_args)
    q2 = torch.linspace(q_limits_low[q_dim1], q_limits_high[q_dim1], N_meshgrid, **robot.tensor_args)
    Q1, Q2 = torch.meshgrid(q1, q2, indexing="ij")

    if q_random is None:
        qs_random = task.sample_q_pos(without_collision=False, n_samples=1)
    else:
        qs_random = q_random.unsqueeze(0)
    # print(qs_random)
    qs_flat = einops.repeat(qs_random, "1 d -> b d", b=Q1.flatten().shape[0]).clone()

    qs_flat[:, q_dim0] = Q1.flatten()
    qs_flat[:, q_dim1] = Q2.flatten()

    Z = task.compute_collision(qs_flat, margin=0)
    valid = Z.reshape(Q1.shape)

    # plot the meshgrid
    if fig is None and ax is None:
        fig, ax = create_fig_and_axes(2)

    cMap = matplotlib.colors.ListedColormap(["white", "grey"])
    if use_imshow:
        ax.imshow(
            ndimage.gaussian_filter(to_numpy(valid).T, sigma=smooth_sigma, order=0),
            cmap=cMap,
            origin="lower",
            alpha=1.0,
            extent=[
                q_limits_low_np[q_dim1],
                q_limits_high_np[q_dim1],
                q_limits_low_np[q_dim0],
                q_limits_high_np[q_dim0],
            ],
            aspect="auto",
        )
    else:
        ax.pcolormesh(
            to_numpy(Q1),
            to_numpy(Q2),
            ndimage.gaussian_filter(to_numpy(valid), sigma=smooth_sigma, order=0),
            cmap=cMap,
            alpha=1.0,
            linewidth=0,
            rasterized=True,
        )

    ax.set_xlabel(f"$q_{q_dim0}$ [rad]")
    ax.set_ylabel(f"$q_{q_dim1}$ [rad]")
    # ax.set_title(f'Configuration free space at q = {to_numpy(qs_random[0])}')

    return fig, ax


if __name__ == "__main__":
    from torch_robotics.tasks.tasks import PlanningTask

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    tensor_args = DEFAULT_TENSOR_ARGS

    env = EnvPlanar2Link(precompute_sdf_obj_fixed=True, sdf_cell_size=0.001, tensor_args=tensor_args)

    robot = RobotPlanar2Link(tensor_args=tensor_args)

    task = PlanningTask(
        env=env,
        robot=robot,
        parametric_trajectory=ParametricTrajectoryBspline(tensor_args=tensor_args),
        ws_limits=torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args),  # workspace limits
        obstacle_cutoff_margin=0.01,
        tensor_args=tensor_args,
    )
    fig, ax = plot_configuration_free_space(task)
    fig.tight_layout()
    plt.show()

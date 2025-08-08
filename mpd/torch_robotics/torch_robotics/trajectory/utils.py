import einops
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import interpolate
from torch.nn import functional as Functional

from torch_robotics.torch_utils.torch_utils import to_torch, to_numpy, DEFAULT_TENSOR_ARGS


def smoothen_trajectory(
    traj_pos,
    n_support_points=30,
    dt=0.02,
    set_average_velocity=True,
    zero_velocity=False,
    tensor_args=DEFAULT_TENSOR_ARGS,
):
    assert not (set_average_velocity and zero_velocity), "Either sets the average velocity or zero velocity"
    traj_pos = to_numpy(traj_pos)
    try:
        # bc_type='clamped' for zero velocities at start and finish
        spline_pos = interpolate.make_interp_spline(
            np.linspace(0, 1, traj_pos.shape[0]), traj_pos, k=3, bc_type="clamped"
        )
        spline_vel = spline_pos.derivative(1)
    except:
        # Trajectory is too short to interpolate, so add last position again and interpolate
        traj_pos = np.vstack((traj_pos, traj_pos[-1] + np.random.normal(0, 0.01)))
        return smoothen_trajectory(
            traj_pos,
            n_support_points=n_support_points,
            dt=dt,
            set_average_velocity=set_average_velocity,
            zero_velocity=zero_velocity,
            tensor_args=tensor_args,
        )

    pos = spline_pos(np.linspace(0, 1, n_support_points))
    vel = np.zeros_like(pos)
    if zero_velocity:
        pass
    elif set_average_velocity:
        avg_vel = (traj_pos[1] - traj_pos[0]) / (n_support_points * dt)
        vel[1:-1, :] = avg_vel
    else:
        vel = spline_vel(np.linspace(0, 1, n_support_points))

    return to_torch(pos, **tensor_args), to_torch(vel, **tensor_args)


def interpolate_traj_via_points(trajs, num_interpolation=10):
    # Interpolates a trajectory linearly between waypoints
    H, D = trajs.shape[-2:]
    if num_interpolation > 0:
        assert trajs.ndim > 1
        traj_dim = trajs.shape
        alpha = torch.linspace(0, 1, num_interpolation + 2).type_as(trajs)[1 : num_interpolation + 1]
        alpha = alpha.view((1,) * len(traj_dim[:-1]) + (-1, 1))
        interpolated_trajs = trajs[..., 0 : traj_dim[-2] - 1, None, :] * alpha + trajs[
            ..., 1 : traj_dim[-2], None, :
        ] * (1 - alpha)
        interpolated_trajs = interpolated_trajs.view(traj_dim[:-2] + (-1, D))
    else:
        interpolated_trajs = trajs
    return interpolated_trajs


def finite_difference_vector(x, dt=1.0, method="forward"):
    # finite differences with zero paddings at the borders
    diff_vector = torch.zeros_like(x)
    if method == "forward":
        diff_vector[..., :-1, :] = torch.diff(x, dim=-2) / dt
    elif method == "backward":
        diff_vector[..., 1:, :] = (x[..., 1:, :] - x[..., :-1, :]) / dt
    elif method == "central":
        diff_vector[..., 1:-1, :] = (x[..., 2:, :] - x[..., :-2, :]) / (2 * dt)
    else:
        raise NotImplementedError
    return diff_vector


def interpolate_points_v1(
    points: torch.Tensor,
    num_interpolated_points: int,
):
    # Interpolates the dimension -2 of the points tensor to have num_interpolated_points, and makes sure that the
    # original points are included.
    # points (batch, horizon, dim) or (batch, horizon, num_points, dim)
    assert points.ndim >= 3, "Points tensor must have at least 3 dimensions"
    *_, h, d = points.shape
    if num_interpolated_points <= h:
        # If we need fewer or the same number of points, just return the original
        return points

    # Create indices for the original points in the interpolated sequence
    original_indices = torch.linspace(0, num_interpolated_points - 1, h).long()

    # Perform linear interpolation
    points_tmp = einops.rearrange(points, "... n d -> (...) n d")
    interpolated_tmp = Functional.interpolate(
        points_tmp.transpose(-2, -1), size=num_interpolated_points, mode="linear", align_corners=True
    ).transpose(-2, -1)
    interpolated_points = interpolated_tmp.view(*points.shape[:-2], num_interpolated_points, d)

    # Ensure original points are included
    interpolated_points[..., original_indices, :] = points

    return interpolated_points


if __name__ == "__main__":
    n_points = 13
    n_interpolated_points = 128
    x = torch.linspace(0, 7, n_points)[..., None] + torch.randn(n_points, 2) * 0.5
    x = x.unsqueeze(0)

    x_interpolated_v1 = interpolate_points_v1(x, n_interpolated_points)

    fig, ax = plt.subplots()
    ax.plot(x[0, :, 0], x[0, :, 1], "ro", label="Original Points", zorder=10)
    ax.plot(x_interpolated_v1[0, :, 0], x_interpolated_v1[0, :, 1], "b-", label="Interpolated Points", marker="x")
    ax.legend()
    plt.show()

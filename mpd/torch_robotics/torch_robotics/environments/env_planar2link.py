import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiSphereField, ObjectField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvPlanar2Link(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        circles = np.array(
            [
                (0.32, 0.3, 0.1),
                (0.37, -0.1, 0.1),
                (0.32, -0.5, 0.15),
                (-0.32, 0.3, 0.20),
                (-0.40, -0.4, 0.25),
            ]
        )
        spheres = MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)
        obj_field = ObjectField([spheres], "planar2link-spheres")
        obj_list = [obj_field]

        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=15)
        if isinstance(robot, tr_robots.RobotPlanar2Link):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=128,
            dt=0.02,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.2,
            pos_only=False,
        )

        if isinstance(robot, tr_robots.RobotPlanar2Link):
            return params
        else:
            raise NotImplementedError


class EnvPlanar2LinkExtraObjectsV00(EnvPlanar2Link):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        # x, y, radius
        circles = np.array(
            [
                (0.05, 0.35, 0.1),
                (-0.40, 0.00, 0.1),
            ]
        )
        obj_extra_list = [MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)]
        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "planar2link-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvPlanar2Link(precompute_sdf_obj_fixed=True, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    ##################################
    env = EnvPlanar2LinkExtraObjectsV00(precompute_sdf_obj_fixed=True, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

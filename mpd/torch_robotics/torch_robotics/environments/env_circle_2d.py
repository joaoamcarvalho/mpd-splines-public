import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvCircle2D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):

        obj_list = [
            MultiSphereField(
                np.array(
                    [
                        [0, 0],
                    ]
                ),
                np.array(
                    [
                        0.3,
                    ]
                ),
                tensor_args=tensor_args,
            ),
            # MultiBoxField(
            #     np.array(
            #         [[0, 0],
            #          ]),
            #     np.array(
            #         [[0.4, 0.4]
            #          ]
            #     )
            #     ,
            #     tensor_args=tensor_args
            # ),
        ]

        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, "dense2d")],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=50)

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                "delta": 1e-2,
                "trust_region": True,
                "method": "cholesky",
            },
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError


if __name__ == "__main__":
    env = EnvCircle2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

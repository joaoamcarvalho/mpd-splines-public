import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvDense2D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):

        obj_list = [
            MultiSphereField(
                np.array(
                    [
                        [-0.43378472328186035, 0.3334643840789795],
                        [0.3313474655151367, 0.6288051009178162],
                        [-0.5656964778900146, -0.484994500875473],
                        [0.42124247550964355, -0.6656165719032288],
                        [0.05636655166745186, -0.5149664282798767],
                        [-0.36961784958839417, -0.12315540760755539],
                        [-0.8740217089653015, -0.4034936726093292],
                        [-0.6359214186668396, 0.6683124899864197],
                        [0.808782160282135, 0.5287870168685913],
                        [-0.023786112666130066, 0.4590069353580475],
                        [0.11544948071241379, -0.12676022946834564],
                        [0.1455741971731186, 0.16420497000217438],
                        [0.628413736820221, -0.43461447954177856],
                        [0.17965620756149292, -0.8926276564598083],
                        [0.6775968670845032, 0.8817358016967773],
                        [-0.3608766794204712, 0.8313458561897278],
                    ]
                ),
                np.array(
                    [
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                        0.125,
                    ]
                ),
                tensor_args=tensor_args,
            ),
            MultiBoxField(
                np.array(
                    [
                        [0.607781708240509, 0.19512386620044708],
                        [0.5575312972068787, 0.5508843064308167],
                        [-0.3352295458316803, -0.6887519359588623],
                        [-0.6572632193565369, 0.31827881932258606],
                        [-0.664594292640686, -0.016457155346870422],
                        [0.8165988922119141, -0.19856023788452148],
                        [-0.8222246170043945, -0.6448580026626587],
                        [-0.2855989933013916, -0.36841487884521484],
                        [-0.8946458101272583, 0.8962447643280029],
                        [-0.23994405567646027, 0.6021060943603516],
                        [-0.006193588487803936, 0.8456171751022339],
                        [0.305103600025177, -0.3661990463733673],
                        [-0.10704007744789124, 0.1318950206041336],
                        [0.7156378626823425, -0.6923345923423767],
                    ]
                ),
                np.array(
                    [
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                        [0.20000000298023224, 0.20000000298023224],
                    ]
                ),
                tensor_args=tensor_args,
            ),
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


class EnvDense2DExtraObjects(EnvDense2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [-0.4, 0.1],
                        [-0.075, -0.85],
                        [-0.1, -0.1],
                    ]
                ),
                np.array(
                    [
                        0.075,
                        0.1,
                        0.075,
                    ]
                ),
                tensor_args=tensor_args,
            ),
            MultiBoxField(
                np.array(
                    [
                        [0.45, -0.1],
                        [0.35, 0.35],
                        [-0.6, -0.85],
                        [-0.65, -0.25],
                    ]
                ),
                np.array(
                    [
                        [0.2, 0.2],
                        [0.1, 0.15],
                        [0.1, 0.25],
                        [0.15, 0.1],
                    ]
                ),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "dense2d-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvDense2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    env = EnvDense2DExtraObjects(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, get_torch_device
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def create_3d_rectangles_objects(tensor_args=DEFAULT_TENSOR_ARGS):
    rectangles_bl_tr = list()
    # bottom left xy, top right xy coordinates
    halfside = 0.2
    rectangles_bl_tr.append((-0.75, -1.0, -0.75 + halfside, -0.3))
    rectangles_bl_tr.append((-0.75, 0.3, -0.75 + halfside, 1.0))
    rectangles_bl_tr.append((-0.75, -0.15, -0.75 + halfside, 0.15))

    rectangles_bl_tr.append((-halfside / 2 - 0.2, -1, halfside / 2 - 0.2, -1 + 0.05))
    rectangles_bl_tr.append((-halfside / 2 - 0.2, -1 + 0.2, halfside / 2 - 0.2, 1 - 0.2))
    rectangles_bl_tr.append((-halfside / 2 - 0.2, 1 - 0.05, halfside / 2 - 0.2, 1))

    rectangles_bl_tr.append((-halfside / 2 + 0.2, -1.0, halfside / 2 + 0.2, -0.3))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, 0.3, halfside / 2 + 0.2, 1.0))
    rectangles_bl_tr.append((-halfside / 2 + 0.2, -0.15, halfside / 2 + 0.2, 0.15))

    rectangles_bl_tr.append((0.75 - halfside, -1, 0.75, -0.6))
    rectangles_bl_tr.append((0.75 - halfside, -0.5, 0.75, -0.2))
    rectangles_bl_tr.append((0.75 - halfside, -0.1, 0.75, 0.1))
    rectangles_bl_tr.append((0.75 - halfside, 0.2, 0.75, 0.5))
    rectangles_bl_tr.append((0.75 - halfside, 0.6, 0.75, 1))

    centers = []
    sizes = []
    for rectangle in rectangles_bl_tr:
        bl_x, bl_y, tr_x, tr_y = rectangle
        x = bl_x + abs(tr_x - bl_x) / 2
        y = bl_y + abs(tr_y - bl_y) / 2
        z = 0
        w = abs(tr_x - bl_x)
        h = abs(tr_y - bl_y)
        d = 1.95
        centers.append((x, y, z))
        sizes.append((w, h, d))

    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    obj_field = ObjectField([boxes], "boxes")
    obj_list = [obj_field]
    return obj_list


class EnvMazeBoxes3D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_list = create_3d_rectangles_objects(tensor_args=tensor_args)

        super().__init__(
            limits=torch.tensor([[-1, -1, -1], [1, 1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=0.01, n_radius=0.3, n_pre_samples=50000, max_time=30)
        if isinstance(robot, tr_robots.RobotPointMass2D):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=100,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=5e-1,
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


if __name__ == "__main__":
    DEFAULT_TENSOR_ARGS["device"] = get_torch_device("cpu")
    env = EnvMazeBoxes3D(precompute_sdf_obj_fixed=True, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    plt.show()

from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


def create_table_object_field(tensor_args=DEFAULT_TENSOR_ARGS):
    centers = [(0.0, 0.0, 0.0)]
    sizes = [(0.56, 0.90, 0.80)]
    centers = np.array(centers)
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], "table")


def create_shelf_field(tensor_args=DEFAULT_TENSOR_ARGS):
    width = 0.80
    height = 2.05
    depth = 0.28
    side_panel_width = 0.02

    shelf_width = width - 2 * side_panel_width
    shelf_height = 0.015
    shelf_depth = depth

    # left panel
    centers = [(side_panel_width / 2, depth / 2, height / 2)]
    sizes = [(side_panel_width, depth, height)]
    # right panel
    centers.append((side_panel_width + shelf_width + side_panel_width / 2, depth / 2, height / 2))
    sizes.append((side_panel_width, depth, height))
    # back panel
    centers.append((side_panel_width + shelf_width / 2, depth + side_panel_width / 2, height / 2))
    sizes.append((shelf_width, side_panel_width, height))

    # bottom shelf
    centers.append((side_panel_width + shelf_width / 2, depth / 2, shelf_height / 2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # top shelf
    centers.append((side_panel_width + shelf_width / 2, depth / 2, height - shelf_height / 2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # shelf 1
    first_shelf_height = 0.82
    centers.append((side_panel_width + shelf_width / 2, depth / 2, first_shelf_height + shelf_height / 2))
    sizes.append((shelf_width, shelf_depth, shelf_height))

    # next shelves
    plus_height_l = [0.23, 0.255, 0.225, 0.225]
    for plus_height in plus_height_l:
        center = list(copy(centers[-1]))
        center[-1] += plus_height
        centers.append(center)
        sizes.append((shelf_width, shelf_depth, shelf_height))

    centers = np.array(centers)
    # main_center = np.array((side_panel_width + shelf_width/2, depth/2, height/2))
    # centers -= main_center
    sizes = np.array(sizes)
    boxes = MultiBoxField(centers, sizes, tensor_args=tensor_args)
    return ObjectField([boxes], "shelf")


class EnvTableShelf(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        # table object field
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table_obj_field.fields[0].sizes[0]
        dist_robot_to_table = 0.0
        theta = np.deg2rad(90)
        ori = np.eye(3)
        ori[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        table_obj_field.set_position_orientation(
            pos=np.array([dist_robot_to_table + table_sizes[1].item() / 2, 0, -table_sizes[2].item() / 2]), ori=ori
        )

        # shelf object field
        shelf_obj_field = create_shelf_field(tensor_args=tensor_args)
        # theta = np.deg2rad(-90)
        dist_table_shelf = 0.15
        shelf_obj_field.set_position_orientation(
            pos=(dist_robot_to_table, dist_table_shelf + table_sizes[0].item() / 2, -table_sizes[2].item()),
            # ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
        )

        obj_list = [table_obj_field, shelf_obj_field]

        super().__init__(
            limits=torch.tensor([[-1, -1, -1], [1.5, 1.0, 1.5]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs,
        )

    def get_gpmp2_params(self, robot=None):
        params = dict(
            opt_iters=250,
            num_samples=64,
            sigma_start=1e-3,
            sigma_gp=1e-1,
            sigma_goal_prior=1e-3,
            sigma_coll=1e-4,
            step_size=5e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.1,
            sigma_start_sample=1e-3,
            sigma_goal_sample=1e-3,
            solver_params={
                "delta": 1e-2,
                "trust_region": True,
                "method": "cholesky",
            },
        )
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        else:
            raise NotImplementedError

    def get_rrt_connect_params(self, robot=None):
        params = dict(n_iters=10000, step_size=torch.pi / 80, n_radius=torch.pi / 4, n_pre_samples=50000, max_time=15)
        if isinstance(robot, tr_robots.RobotPanda):
            return params
        else:
            raise NotImplementedError


class EnvTableShelfExtraObjectsV00(EnvTableShelf):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.5, 0.2, 0.25 / 2],
                    ]
                ),
                np.array(
                    [
                        [0.25, 0.25, 0.25],
                    ]
                ),
                tensor_args=tensor_args,
            )
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "tableshelf-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvTableShelf(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    # env.render_grad_sdf(ax, fig)
    plt.show()

    ############################
    env = EnvTableShelfExtraObjectsV00(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

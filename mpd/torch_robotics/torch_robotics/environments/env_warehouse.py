from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.env_table_shelf import create_table_object_field, create_shelf_field
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvWarehouse(EnvBase):

    def __init__(self, rotation_z_axis_deg: float = 0, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):  # degrees
        rotation_z_axis_rad = np.deg2rad(rotation_z_axis_deg)
        H_perturbation = np.eye(4)
        H_perturbation[:2, :2] = np.array(
            [
                [np.cos(rotation_z_axis_rad), -np.sin(rotation_z_axis_rad)],
                [np.sin(rotation_z_axis_rad), np.cos(rotation_z_axis_rad)],
            ]
        )
        H_perturbation_th = to_torch(H_perturbation, **tensor_args)

        # table object field
        table_obj_field = create_table_object_field(tensor_args=tensor_args)
        table_sizes = table_obj_field.fields[0].sizes[0]
        dist_robot_to_table = 0.0
        theta = np.deg2rad(90)
        ori = np.eye(3)
        ori[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H = np.eye(4)
        H[:3, :3] = ori
        H[:3, 3] = np.array([dist_robot_to_table + table_sizes[1].item() / 2, 0, -table_sizes[2].item() / 2])
        H = H_perturbation @ H
        table_obj_field.set_position_orientation(pos=H[:3, 3], ori=H[:3, :3])

        # shelf object field
        shelf_obj_field = create_shelf_field(tensor_args=tensor_args)
        dist_table_shelf = 0.15
        theta = np.deg2rad(0)
        ori = np.eye(3)
        ori[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H = np.eye(4)
        H[:3, :3] = ori
        H[:3, 3] = np.asarray(
            (dist_robot_to_table, dist_table_shelf + table_sizes[0].item() / 2, -table_sizes[2].item())
        )
        H = H_perturbation @ H
        shelf_obj_field.set_position_orientation(pos=H[:3, 3], ori=H[:3, :3])

        # small shelf field
        small_shelf_sizes = [0.8, 0.28, 1.06]
        small_shelf_object = MultiBoxField(
            np.array([[0.0, 0.0, 0.0]]), np.array([small_shelf_sizes]), tensor_args=tensor_args
        )
        small_shelf_object_field = ObjectField([small_shelf_object], "small_shelf")
        theta = np.deg2rad(0)
        ori = np.eye(3)
        ori[:2, :2] = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        H = np.eye(4)
        H[:3, :3] = ori
        H[:3, 3] = np.asarray(
            (
                dist_robot_to_table + small_shelf_sizes[0] / 2,
                -dist_table_shelf - table_sizes[1].item() / 2,
                -small_shelf_sizes[2] / 2 + (small_shelf_sizes[2] - table_sizes[2].item()),
            )
        )
        H = H_perturbation @ H
        small_shelf_object_field.set_position_orientation(pos=H[:3, 3], ori=H[:3, :3])

        obj_list = [table_obj_field, shelf_obj_field, small_shelf_object_field]

        if "obj_extra_list" in kwargs:
            for obj in kwargs["obj_extra_list"]:
                H = H_perturbation_th @ obj.get_transformation_matrix()
                obj.set_position_orientation(pos=H[:3, 3], ori=H[:3, :3])

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


class EnvWarehouseExtraObjectsV00(EnvWarehouse):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.85, 0.1, 0.25 / 2],
                        [0.6, -0.15, 0.5 / 2],
                    ]
                ),
                np.array(
                    [
                        [0.1, 0.25, 0.25],
                        [0.25, 0.25, 0.5],
                    ]
                ),
                tensor_args=tensor_args,
            )
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "tableshelf-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvWarehouse(rotation_z_axis_deg=-45, tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    ################################
    env = EnvWarehouseExtraObjectsV00(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

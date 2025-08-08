import os

import numpy as np
from colour import Color
from matplotlib.colors import to_rgb

from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS


class RobotPlanar2Link(RobotBase):

    link_name_ee = "link_ee"  # must be in the urdf file

    def __init__(
        self,
        urdf_robot_file=os.path.join(get_robot_path(), "planar_robot", "planar_robot_2_link.urdf"),
        collision_spheres_file_path=os.path.join(
            get_configs_path(), "planar_robot_2_link/planar_robot_2_link_sphere_config.yaml"
        ),
        joint_limits_file_path=os.path.join(get_configs_path(), "planar_robot_2_link/joint_limits.yaml"),
        task_space_dim=2,
        **kwargs,
    ):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            joint_limits_file_path=joint_limits_file_path,
            link_name_ee=self.link_name_ee,
            task_space_dim=task_space_dim,
            **kwargs,
        )

        ################################################################################################
        # Link indices for rendering
        self.link_idxs = self.get_link_idxs_for_rendering()

    def get_link_idxs_for_rendering(self):
        return [self.robot_torchkin.link_map["link_2"].id, self.robot_torchkin.link_map[self.link_name_ee].id]

    def render(self, ax, q=None, alpha=1.0, color="blue", linewidth=2.0, zorder=1, ee_size=10, **kwargs):
        H_all = self.fk_all(q.unsqueeze(0))
        p_all = [link_pos_from_link_tensor(H_all[idx]).squeeze() for idx in self.link_idxs]
        p_all = to_numpy([to_numpy(p) for p in p_all])
        ax.plot(
            [0, p_all[0][0]],
            [0, p_all[0][1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
            solid_capstyle="round",
        )
        for p1, p2 in zip(p_all[:-1], p_all[1:]):
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
                solid_capstyle="round",
            )
        # render end-effector
        ax.scatter(p_all[-1][0], p_all[-1][1], color=color, marker="o", zorder=zorder + 10, s=ee_size**2.2, alpha=alpha)

    def render_trajectories(
        self,
        ax,
        q_pos_trajs=None,
        start_state=None,
        goal_state=None,
        colors=["gray"],
        n_skip_points=None,
        ee_goal_position=None,
        alpha_trajectory=0.5,
        **kwargs,
    ):
        if q_pos_trajs is not None:
            trajs_pos = self.get_position(q_pos_trajs)
            if trajs_pos.ndim == 2:
                trajs_pos = trajs_pos.unsqueeze(0)
            for _trajs_pos, color in zip(trajs_pos, colors):
                # skip some points for visualization
                _trajs_pos = _trajs_pos[::n_skip_points] if n_skip_points is not None else _trajs_pos
                for q in _trajs_pos:
                    self.render(ax, q, alpha=0.5, color=color, zorder=2, **kwargs)
        if start_state is not None:
            self.render(ax, start_state, alpha=1.0, color="blue", zorder=100, **kwargs)
        if goal_state is not None:
            self.render(ax, goal_state, alpha=1.0, color="red", zorder=100, **kwargs)
        elif q_pos_trajs is not None:
            self.render(ax, trajs_pos[0][-1], alpha=1.0, color="magenta", zorder=10, **kwargs)
        if ee_goal_position is not None:
            ee_goal_position_np = to_numpy(ee_goal_position)
            ax.scatter(ee_goal_position_np[0], ee_goal_position_np[1], color="red", marker="*", s=10**2.6, zorder=100)


class RobotPlanar4Link(RobotPlanar2Link):

    def __init__(
        self,
        urdf_robot_file=os.path.join(get_robot_path(), "planar_robot", "planar_robot_4_link.urdf"),
        collision_spheres_file_path=os.path.join(
            get_configs_path(), "planar_robot_4_link/planar_robot_4_link_sphere_config.yaml"
        ),
        joint_limits_file_path=os.path.join(get_configs_path(), "planar_robot_4_link/joint_limits.yaml"),
        **kwargs,
    ):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            joint_limits_file_path=joint_limits_file_path,
            **kwargs,
        )

    def get_link_idxs_for_rendering(self):
        return [
            self.robot_torchkin.link_map["link_2"].id,
            self.robot_torchkin.link_map["link_3"].id,
            self.robot_torchkin.link_map["link_4"].id,
            self.robot_torchkin.link_map[self.link_name_ee].id,
        ]


if __name__ == "__main__":
    robot = RobotPlanar2Link(tensor_args=DEFAULT_TENSOR_ARGS)
    robot = RobotPlanar4Link(tensor_args=DEFAULT_TENSOR_ARGS)

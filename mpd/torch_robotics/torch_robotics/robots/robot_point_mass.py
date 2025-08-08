import os.path

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torch_robotics.environments.primitives import plot_sphere
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path, get_configs_path
from torch_robotics.torch_utils.torch_utils import to_numpy


class RobotPointMass2D(RobotBase):

    link_name_ee = "robot"  # must be in the urdf file

    def __init__(
        self,
        urdf_robot_file=os.path.join(get_robot_path(), "point_mass", "point_mass_robot_2d.urdf"),
        collision_spheres_file_path=os.path.join(
            get_configs_path(), "point_mass_robot_2d/point_mass_robot_2d_sphere_config.yaml"
        ),
        task_space_dim=2,
        **kwargs,
    ):

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=collision_spheres_file_path,
            link_name_ee=self.link_name_ee,
            task_space_dim=task_space_dim,
            **kwargs,
        )

    def render(self, ax, q_pos=None, color="blue", cmap="Blues", margin_multiplier=1.0, **kwargs):
        if q_pos is not None:
            margin = self.link_collision_spheres_radii[0] * margin_multiplier
            q_pos = to_numpy(q_pos)
            if q_pos.ndim == 1:
                if self.q_dim == 2:
                    circle1 = plt.Circle(q_pos, margin, color=color, zorder=10)
                    ax.add_patch(circle1)
                elif self.q_dim == 3:
                    plot_sphere(ax, q_pos, np.zeros_like(q_pos), margin, cmap)
                else:
                    raise NotImplementedError
            elif q_pos.ndim == 2:
                if q_pos.shape[-1] == 2:
                    # ax.scatter(q[:, 0], q[:, 1], color=color, s=10 ** 2, zorder=10)
                    circ = []
                    for q_ in q_pos:
                        circ.append(plt.Circle(q_, margin, color=color))
                        coll = mcoll.PatchCollection(circ, zorder=10)
                        ax.add_collection(coll)
                elif q_pos.shape[-1] == 3:
                    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], color=color, s=10 ** 2, zorder=10)
                    for q_ in q_pos:
                        plot_sphere(ax, q_, np.zeros_like(q_), margin, cmap)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def render_trajectories(
        self,
        ax,
        q_pos_trajs=None,
        q_pos_start=None,
        q_pos_goal=None,
        colors=["blue"],
        linestyle="solid",
        control_points=None,
        plot_points_scatter=False,
        line_alpha=1.0,
        plot_robot_collision_sphere=False,
        **kwargs,
    ):
        if q_pos_trajs is not None:
            trajs_pos = self.get_position(q_pos_trajs)
            trajs_np = to_numpy(trajs_pos)
            if self.q_dim == 3:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1], trajs_np[..., 2]))).swapaxes(1, 2)
                line_segments = Line3DCollection(segments, colors=colors, linestyle=linestyle)
                ax.add_collection(line_segments)
                points = np.reshape(trajs_np, (-1, 3))
                colors_scatter = []
                for segment, color in zip(segments, colors):
                    colors_scatter.extend([color] * segment.shape[0])
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=colors_scatter, s=2**2)
            else:
                segments = np.array(list(zip(trajs_np[..., 0], trajs_np[..., 1]))).swapaxes(1, 2)
                line_segments = mcoll.LineCollection(segments, colors=colors, linestyle=linestyle, alpha=line_alpha)
                ax.add_collection(line_segments)
                if control_points is not None:
                    points = np.reshape(to_numpy(control_points), (-1, 2))
                    colors_scatter = ["blue"] * points.shape[0]
                    # for control_points_aux, color in zip(control_points, colors):
                    #     colors_scatter.extend([color]*control_points_aux.shape[0])
                else:
                    points = np.reshape(trajs_np, (-1, 2))
                    colors_scatter = ["red"] * points.shape[0]
                    # colors_scatter = []
                    # for segment, color in zip(segments, colors):
                    #     colors_scatter.extend([color]*segment.shape[0])
                if plot_points_scatter:
                    ax.scatter(points[:, 0], points[:, 1], color=colors_scatter, s=2**2, zorder=100)

                if plot_robot_collision_sphere and self.link_collision_spheres_radii.ndim == 1:
                    fig_width_inches = ax.get_figure().get_figwidth()
                    points_per_unit = fig_width_inches * 72 / 2  # 72 points per inch, range is 2 units
                    marker_size = (
                        to_numpy(self.link_collision_spheres_radii) * 2 * points_per_unit
                    ) ** 2  # diameter = 2 * radius
                    ax.scatter(
                        trajs_np[..., 0].reshape(-1),
                        trajs_np[..., 1].reshape(-1),
                        alpha=0.25,
                        s=marker_size,
                        color="blue",
                    )

        if q_pos_start is not None:
            q_pos_start_np = to_numpy(q_pos_start)
            if len(q_pos_start_np) == 3:
                ax.plot(q_pos_start_np[0], q_pos_start_np[1], q_pos_start_np[2], "go", markersize=7)
            else:
                ax.plot(q_pos_start_np[0], q_pos_start_np[1], "go", markersize=7)
        if q_pos_goal is not None:
            q_pos_goal_np = to_numpy(q_pos_goal)
            if len(q_pos_goal_np) == 3:
                ax.plot(q_pos_goal_np[0], q_pos_goal_np[1], q_pos_goal_np[2], marker="o", color="purple", markersize=7)
            else:
                ax.plot(q_pos_goal_np[0], q_pos_goal_np[1], marker="o", color="purple", markersize=7)


# alias for backward compatibility
RobotPointMass = RobotPointMass2D


class RobotPointMass3D(RobotPointMass2D):

    def __init__(self, **kwargs):
        super().__init__(
            urdf_robot_file=os.path.join(get_robot_path(), "point_mass", "point_mass_robot_3d.urdf"),
            collision_spheres_file_path=os.path.join(
                get_configs_path(), "point_mass_robot_3d/point_mass_robot_3d_sphere_config.yaml"
            ),
            task_space_dim=3,
            **kwargs,
        )

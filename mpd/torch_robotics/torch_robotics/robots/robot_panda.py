import os

from torch_robotics.environments.primitives import MultiSphereField
from torch_robotics.robots.robot_base import RobotBase
from torch_robotics.torch_kinematics_tree.geometrics.skeleton import get_skeleton_from_model
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robot_tree import convert_link_dict_to_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_kinematics_tree.utils.files import get_configs_path, get_robot_path
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import plot_coordinate_frame, create_fig_and_axes


class RobotPanda(RobotBase):

    link_name_ee = "panda_hand"  # must be in the urdf file

    def __init__(self, gripper=False, grasped_object=None, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):

        if gripper:
            urdf_robot_file = os.path.join(
                get_robot_path(), "franka_description", "robots", "panda_arm_hand_fixed_gripper.urdf"
            )
        else:
            urdf_robot_file = os.path.join(
                get_robot_path(), "franka_description", "robots", "panda_arm_hand_no_gripper.urdf"
            )

        ##########################################################################################
        super().__init__(
            urdf_robot_file=urdf_robot_file,
            collision_spheres_file_path=os.path.join(get_configs_path(), "panda/panda_sphere_config.yaml"),
            joint_limits_file_path=os.path.join(get_configs_path(), "panda/joint_limits.yaml"),
            link_name_ee=self.link_name_ee,
            gripper_q_dim=0 if gripper else 0,  # the gripper is fixed
            grasped_object=grasped_object,
            tensor_args=tensor_args,
            **kwargs,
        )

        ##########################################################################################
        # Differentiable franka panda model for visualization
        self.diff_panda = DifferentiableFrankaPanda(
            gripper=gripper, device=tensor_args["device"], grasped_object=grasped_object
        )

    def render(
        self,
        ax,
        q=None,
        color="blue",
        arrow_length=0.15,
        arrow_alpha=1.0,
        arrow_linewidth=2.0,
        draw_links_spheres=False,
        **kwargs,
    ):
        # draw skeleton
        skeleton = get_skeleton_from_model(self.diff_panda, q, self.diff_panda.get_link_names())
        skeleton.draw_skeleton(ax=ax, color=color)

        # forward kinematics
        fks_dict = self.diff_panda.compute_forward_kinematics_all_links(q.unsqueeze(0), return_dict=True)

        # draw link collision points
        if draw_links_spheres:
            link_tensor = convert_link_dict_to_tensor(fks_dict, self.link_collision_spheres_names)
            link_pos = link_pos_from_link_tensor(link_tensor)
            spheres = MultiSphereField(
                link_pos.squeeze(0), self.link_collision_spheres_radii.view(-1, 1), tensor_args=self.tensor_args
            )
            spheres.render(ax, color="red", cmap="Reds", **kwargs)

        # draw EE frame
        frame_EE = fks_dict[self.link_name_ee]
        plot_coordinate_frame(
            ax,
            frame_EE,
            tensor_args=self.tensor_args,
            arrow_length=arrow_length,
            arrow_alpha=arrow_alpha,
            arrow_linewidth=arrow_linewidth,
        )

    def render_trajectories(self, ax, q_pos_trajs=None, start_state=None, goal_state=None, colors=["gray"], **kwargs):
        if q_pos_trajs is not None:
            trajs_pos = self.get_position(q_pos_trajs)
            for traj, color in zip(trajs_pos, colors):
                for t in range(traj.shape[0]):
                    q = traj[t]
                    self.render(ax, q, color, **kwargs, arrow_length=0.1, arrow_alpha=0.5, arrow_linewidth=1.0)
            if start_state is not None:
                self.render(ax, start_state, color="green")
            if goal_state is not None:
                self.render(ax, goal_state, color="purple")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch

    robot = RobotPanda(tensor_args=DEFAULT_TENSOR_ARGS)

import os
import time
from copy import copy
from math import ceil
from pathlib import Path

import cv2
import imageio
import matplotlib
import numpy as np
from dotmap import DotMap
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from moviepy.video.io.VideoFileClip import VideoFileClip

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.gymutil import parse_device_str
from isaacgym.torch_utils import *

import torch

from deps.isaacgym.python.isaacgym.torch_utils import to_torch
from mpd.parametric_trajectory.trajectory_bspline import ParametricTrajectoryBspline
from torch_robotics.environments import EnvTableShelf, EnvWarehouse
from torch_robotics.environments.env_spheres_3d import EnvSpheres3D
from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField
from torch_robotics.robots.robot_panda import RobotPanda
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import rotation_matrix_to_q
from torch_robotics.torch_kinematics_tree.models.robots import modify_robot_urdf_grasped_object
from torch_robotics.torch_kinematics_tree.utils.files import get_robot_path
from torch_robotics.trajectory.utils import interpolate_points_v1
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device, to_numpy


GYM_COLORS = {
    "purple": gymapi.Vec3(128 / 255.0, 0.0, 128 / 255.0),
    "black": gymapi.Vec3(0.0, 0.0, 0.0),
    "grey": gymapi.Vec3(220.0 / 255.0, 220.0 / 255.0, 220.0 / 255.0),
    "red": gymapi.Vec3(1.0, 0.0, 0.0),
}


def set_object_position_and_orientation(obj_pos, obj_ori):
    # set position and orientation
    obj_pose = gymapi.Transform()
    obj_pose.p = gymapi.Vec3(*obj_pos)
    obj_pose.r = gymapi.Quat(obj_ori[1], obj_ori[2], obj_ori[3], obj_ori[0])
    return obj_pose


def create_isaac_assets_from_primitive_shapes(sim, gym, obj_list):
    object_assets_l = []
    object_poses_l = []

    for obj in obj_list or []:
        # get position and orientation
        obj_pos = to_numpy(obj.pos)
        obj_ori_mat = to_numpy(obj.ori)
        obj_ori_quat = to_numpy(rotation_matrix_to_q(obj.ori))

        for obj_field in obj.fields:
            if isinstance(obj_field, MultiSphereField):
                for center, radius in zip(obj_field.centers, obj_field.radii):
                    if center.nelement() == 2:  # add z coordinate
                        center = torch.cat([center, torch.zeros(1, dtype=center.dtype, device=center.device)])
                    center_np = to_numpy(center)
                    radius_np = to_numpy(radius)

                    # create sphere asset
                    sphere_radius = radius_np
                    asset_options = gymapi.AssetOptions()
                    asset_options.fix_base_link = True
                    sphere_asset = gym.create_sphere(sim, sphere_radius, asset_options)
                    object_assets_l.append(sphere_asset)

                    # set position and orientation
                    object_poses_l.append(
                        set_object_position_and_orientation(obj_ori_mat @ center_np + obj_pos, obj_ori_quat)
                    )

            elif isinstance(obj_field, MultiBoxField):
                for center, size in zip(obj_field.centers, obj_field.sizes):
                    if center.nelement() == 2:  # add z coordinate
                        center = torch.cat([center, torch.zeros(1, dtype=center.dtype, device=center.device)])
                        size = torch.cat([size, torch.tensor([0.1], dtype=size.dtype, device=size.device)])
                    center_np = to_numpy(center)
                    size_np = to_numpy(size)

                    # create box asset
                    asset_options = gymapi.AssetOptions()
                    asset_options.fix_base_link = True
                    sphere_asset = gym.create_box(sim, size_np[0], size_np[1], size_np[2], asset_options)
                    object_assets_l.append(sphere_asset)

                    # set position and orientation
                    object_poses_l.append(
                        set_object_position_and_orientation(obj_ori_mat @ center_np + obj_pos, obj_ori_quat)
                    )
            else:
                raise NotImplementedError

    return object_assets_l, object_poses_l


def make_gif_from_array(filename, array, fps=10):
    # https://gist.github.com/nirum/d4224ad3cd0d71bfef6eba8f3d6ffd59
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> make_gif_from_array('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    """
    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_gif(filename, fps=fps)
    return clip


class CameraRecorder:

    def __init__(self, dt=0.01):
        self.dt = dt
        self.step_img = []

    def reset(self):
        self.step_img = []

    def append(self, step, img):
        self.step_img.append((step, img))

    def make_video(
        self,
        video_duration=5.0,
        video_path="./trajs_replay.mp4",
        make_gif=False,
        n_pre_steps=10,
        n_post_steps=10,
        save_step_images=False,
        **kwargs,
    ):
        if len(self.step_img) == 0:
            print("No images to a make video")
            return

        # remove first and last steps from the video
        del self.step_img[:n_pre_steps]
        del self.step_img[-n_post_steps:]

        # set up the video writer
        fps = len(self.step_img) / video_duration
        width, height, layers = self.step_img[0][1].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(video_path, fourcc, fps, (height, width))

        print(f"Making video with {len(self.step_img)} frames")
        max_steps = len(self.step_img) - 1
        image_l = []
        step_text = 0
        for step, frame in self.step_img:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # rotate the frame
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # Add step text to frame
            cv2.putText(
                frame,
                f"Step: {step_text}/{max_steps}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_4,
            )
            cv2.putText(
                frame,
                f"Time: {self.dt * step_text:.2f} secs",
                (50, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_4,
            )

            video.write(frame)
            # save frame for gif
            image_l.append(frame)

            # increment step text
            step_text += 1

        cv2.destroyAllWindows()
        video.release()
        print(f"...finished making video at {video_path}")

        # create gif
        if make_gif:
            video = VideoFileClip(video_path)
            # ensure that the file has the .gif extension
            gif_path, _ = os.path.splitext(video_path)
            gif_path = gif_path + ".gif"
            video.write_gif(gif_path, fps=video.fps)

        if save_step_images:
            video_path_root = video_path.split(".mp4")[0]
            step_images_path = os.path.join(video_path_root, "step_images")
            os.makedirs(step_images_path, exist_ok=True)
            for i, img in enumerate(image_l):
                cv2.imwrite(os.path.join(step_images_path, f"step_{i}.png"), img)


class MotionPlanningIsaacGymEnv:

    def __init__(
        self,
        env,
        robot,
        asset_root=f"{os.environ['HOME']}/Projects/MotionPlanningDiffusion/mpd/deps/isaacgym/assets",
        robot_asset_file="urdf/franka_description/robots/franka_panda.urdf",
        enable_dynamics=False,
        controller_type="position",
        num_envs=8,
        add_ground_plane=True,
        all_robots_in_one_env=False,
        use_gpu_pipeline=False,
        show_viewer=False,
        sync_viewer_with_real_time=False,
        viewer_time_between_steps=0.01,
        color_robots=False,
        collor_robots_in_collision=False,
        draw_goal_configuration=False,
        draw_contact_forces=False,
        draw_end_effector_path=False,
        draw_end_effector_frame=False,
        draw_ee_pose_goal=False,
        render_camera_global=False,
        render_camera_global_append_to_recorder=False,
        camera_global_width=1920,
        camera_global_height=1080,
        # camera_global_width=1280, camera_global_height=720,
        # camera_global_width=640, camera_global_height=480,
        camera_global_from_top=False,
        cam_pos=gymapi.Vec3(1.2, -1.2, 0.9),
        cam_target=gymapi.Vec3(0.1, 0.1, 0.5),
        **kwargs,
    ):

        self.tensor_args = {"device": get_torch_device(device="cuda" if use_gpu_pipeline else "cpu")}

        self.env_tr = env
        self.robot_tr = robot

        self.controller_type = controller_type
        self.enable_dynamics = enable_dynamics

        # create num_envs environments, add 1 for the goal configuration
        self.num_envs = num_envs + 1 if draw_goal_configuration else num_envs
        self.env_idxs = list(range(self.num_envs - 1 if draw_goal_configuration else self.num_envs - 1))
        self.env_goal_configuration_idx = self.num_envs - 1 if draw_goal_configuration else None

        self.all_robots_in_one_env = all_robots_in_one_env

        ###############################################################################################################
        # Visualizations
        self.color_robots = color_robots
        self.collor_robots_in_collision = collor_robots_in_collision
        self.draw_goal_configuration = draw_goal_configuration
        self.draw_ee_pose_goal = draw_ee_pose_goal
        self.goal_joint_position = None
        self.draw_contact_forces = draw_contact_forces
        self.draw_end_effector_path = draw_end_effector_path
        self.draw_end_effector_frame = draw_end_effector_frame
        self.axes_geom = gymutil.AxesGeometry(0.15)
        self.axes_geom_ee_goal = gymutil.AxesGeometry(0.3)
        self.end_effector_positions_visualization = None

        self._ee_pose_goal = None

        ###############################################################################################################
        # ISAAC GYM options
        ###############################################################################################################
        # Setup simulator
        # acquire gym interface
        self.gym = gymapi.acquire_gym()

        # gym arguments
        # self.gym_args = gymutil.parse_arguments()  # we cannot call this function because it parses cmd line arguments

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        # sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
        sim_params.dt = 1 / 60.0
        sim_params.substeps = 2
        sim_params.use_gpu_pipeline = use_gpu_pipeline
        physics_engine = gymapi.SIM_PHYSX
        if physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.rest_offset = 0.0
            sim_params.physx.contact_offset = 0.0001
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = 0
            sim_params.physx.use_gpu = use_gpu_pipeline
        else:
            raise Exception("Can only be used with PhysX")

        # create simulation
        # https://forums.developer.nvidia.com/t/not-sure-what-to-set-for-graphics-device-id/193625/2
        graphics_device_id = 0 if show_viewer or render_camera_global else -1
        sim_device_type, compute_device_id = parse_device_str(self.tensor_args["device"].type)
        self.sim = self.gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

        ###############################################################################################################
        # Environment assets
        object_fixed_assets_l, object_fixed_poses_l = create_isaac_assets_from_primitive_shapes(
            self.sim, self.gym, self.env_tr.obj_fixed_list
        )
        object_extra_assets_l, object_extra_poses_l = create_isaac_assets_from_primitive_shapes(
            self.sim, self.gym, self.env_tr.obj_extra_list
        )

        ###############################################################################################################
        # Robot asset

        # load robot asset
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = True
        robot_asset = self.gym.load_asset(self.sim, os.path.abspath(asset_root), robot_asset_file, asset_options)

        self.robot_end_effector_link_name = self.robot_tr.link_name_ee

        # configure franka dofs
        robot_dof_properties = self.gym.get_asset_dof_properties(robot_asset)
        self.robot_dof_lower_limits = robot_dof_properties["lower"]
        self.robot_dof_upper_limits = robot_dof_properties["upper"]
        robot_dof_ranges = self.robot_dof_upper_limits - self.robot_dof_lower_limits
        robot_dof_mids = 0.3 * (self.robot_dof_upper_limits + self.robot_dof_lower_limits)

        # robot arm properties
        if self.controller_type == "position":
            robot_dof_properties["driveMode"][: self.robot_tr.arm_q_dim].fill(gymapi.DOF_MODE_POS)
            robot_dof_properties["stiffness"][: self.robot_tr.arm_q_dim].fill(1000.0)
            robot_dof_properties["damping"][: self.robot_tr.arm_q_dim] = 2.0 * np.sqrt(
                robot_dof_properties["stiffness"][: self.robot_tr.arm_q_dim]
            )
        else:
            raise NotImplementedError

        # gripper properties
        if self.robot_tr.gripper_q_dim > 0:
            robot_dof_properties["driveMode"][self.robot_tr.arm_q_dim :].fill(gymapi.DOF_MODE_POS)
            robot_dof_properties["stiffness"][self.robot_tr.arm_q_dim :].fill(800.0)
            robot_dof_properties["damping"][self.robot_tr.arm_q_dim :].fill(40.0)

        # default dof states (position and velocities)
        self.robot_num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.default_dof_pos = np.zeros(self.robot_num_dofs, dtype=np.float32)
        # default_dof_pos[:self.robot.arm_q_dim] = robot_dof_mids[:self.robot.arm_q_dim]

        # grippers open
        if self.robot_tr.gripper_q_dim > 0:
            self.default_dof_pos[self.robot_tr.arm_q_dim :] = self.robot_dof_upper_limits[self.robot_tr.arm_q_dim :]

        self.default_dof_state = np.zeros(self.robot_num_dofs, gymapi.DofState.dtype)
        self.default_dof_state["pos"] = self.default_dof_pos

        ###############################################################################################################
        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 1.5
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print(f"Creating {self.num_envs} environments")

        # add ground plane
        if add_ground_plane:
            plane_params = gymapi.PlaneParams()
            plane_params.distance = 2
            plane_params.normal = gymapi.Vec3(0, 0, 1)
            self.gym.add_ground(self.sim, plane_params)

        # robots pose
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0, 0, 0)

        self.envs = []
        self.robot_handles = []
        self.robot_idxs = []
        self.obj_idxs = []
        self.hand_idxs = []

        color_obj_fixed = GYM_COLORS["grey"]
        color_obj_extra = GYM_COLORS["red"]

        # robot colors
        cmap = matplotlib.cm.get_cmap("viridis")
        self.robot_colors = [cmap(m)[:3] for m in np.linspace(0, 1, self.num_envs)]
        if self.color_robots:
            self.sphere_geom_end_effector_l = [
                gymutil.WireframeSphereGeometry(0.005, 10, 10, gymapi.Transform(), color=(c[0], c[1], c[2]))
                for c in self.robot_colors
            ]
        else:
            self.sphere_geom_end_effector_l = [
                gymutil.WireframeSphereGeometry(
                    0.005,
                    10,
                    10,
                    gymapi.Transform(),
                    # color=(0, 0, 1)
                    color=(1, 0.65, 0),
                )
                for _ in self.robot_colors
            ]

        # Maps the global rigid body index to the environments index
        # Useful to know which trajectories are in collision
        self.map_rigid_body_idxs_to_env_idx = {}

        # Create environments
        if self.all_robots_in_one_env:
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

        for i in range(self.num_envs):
            # Create environment
            if not self.all_robots_in_one_env:
                env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
                self.envs.append(env)

            # Add objects fixed
            for obj_asset, obj_pose in zip(object_fixed_assets_l, object_fixed_poses_l):
                object_handle = self.gym.create_actor(env, obj_asset, obj_pose, "obj_fixed", i, 0)
                self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_fixed)
                # get global index of object in rigid body state tensor
                obj_idx = self.gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(obj_idx)
                self.map_rigid_body_idxs_to_env_idx[obj_idx] = i

            # Add objects extra
            for obj_asset, obj_pose in zip(object_extra_assets_l, object_extra_poses_l):
                object_handle = self.gym.create_actor(env, obj_asset, obj_pose, "obj_extra", i, 0)
                self.gym.set_rigid_body_color(env, object_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color_obj_extra)
                # get global index of object in rigid body state tensor
                obj_idx = self.gym.get_actor_rigid_body_index(env, object_handle, 0, gymapi.DOMAIN_SIM)
                self.obj_idxs.append(obj_idx)
                self.map_rigid_body_idxs_to_env_idx[obj_idx] = i

            # Add the robot
            # Set to 0 to enable self-collision.
            # By default, we do not consider self-collision, because the collision meshes are too conservative.
            robot_handle = self.gym.create_actor(env, robot_asset, robot_pose, f"robot_{i}", i, 1)
            self.robot_handles.append(robot_handle)
            self.robot_idxs.append(self.gym.get_actor_index(env, robot_handle, gymapi.DOMAIN_SIM))
            rb_names = self.gym.get_actor_rigid_body_names(env, robot_handle)
            for j in range(len(rb_names)):
                rb_idx = self.gym.get_actor_rigid_body_index(env, robot_handle, j, gymapi.DOMAIN_SIM)
                self.map_rigid_body_idxs_to_env_idx[rb_idx] = i

            # color robot
            n_rigid_bodies = self.gym.get_actor_rigid_body_count(env, robot_handle)
            if self.draw_goal_configuration and i == self.num_envs - 1:
                for j in range(n_rigid_bodies):
                    self.gym.set_rigid_body_color(
                        env, robot_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, GYM_COLORS["purple"]
                    )

            if color_robots:
                if not (self.draw_goal_configuration and i == self.num_envs - 1):
                    c = self.robot_colors[i]
                    color = gymapi.Vec3(c[0], c[1], c[2])
                    for j in range(n_rigid_bodies):
                        self.gym.set_rigid_body_color(env, robot_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, color)

            # set robot dof properties
            self.gym.set_actor_dof_properties(env, robot_handle, robot_dof_properties)

        ###############################################################################################################
        # VIEWER
        self.viewer = None
        if show_viewer:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        self.sync_viewer_with_real_time = sync_viewer_with_real_time
        self.viewer_time_between_steps = viewer_time_between_steps

        ###############################################################################################################
        # CAMERA GLOBAL
        # point camera at middle env
        if camera_global_from_top:
            cam_pos = gymapi.Vec3(1e-3, -1e-3, 1.90)
            cam_target = gymapi.Vec3(0, -1e-3, -1)

        if len(self.envs) == 1:
            self.middle_env = self.envs[0]
        else:
            self.middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
        if self.viewer is not None:
            self.gym.viewer_camera_look_at(self.viewer, self.middle_env, cam_pos, cam_target)

        # camera global for rendering
        self.render_camera_global = render_camera_global
        self.render_camera_global_append_to_recorder = render_camera_global_append_to_recorder
        if self.render_camera_global:
            camera_props = gymapi.CameraProperties()
            camera_props.width, camera_props.height = camera_global_width, camera_global_height
            self.camera_global_handle = self.gym.create_camera_sensor(self.middle_env, camera_props)
            self.gym.set_camera_location(self.camera_global_handle, self.middle_env, cam_pos, cam_target)
        self.camera_global_recorder = CameraRecorder(dt=viewer_time_between_steps)

        ###############################################################################################################
        # ==== Prepare tensors =====
        # use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        # get actors root state tensor
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.initial_root_states = gymtorch.wrap_tensor(_actor_root_state_tensor).clone()
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.robot_dof_states = gymtorch.wrap_tensor(_dof_states).view(self.num_envs, self.robot_num_dofs, 2)
        self.robot_dof_pos = self.robot_dof_states[..., 0]
        self.robot_dof_vel = self.robot_dof_states[..., 1]
        self.gym.refresh_dof_state_tensor(self.sim)

        ###############################################################################################################
        # Simulation step counter
        self.step_idx = 0

    def reset(self, q_pos_starts=None, q_pos_goal=None):
        # Reset step counter
        self.step_idx = 0

        # reset camera recorder
        self.camera_global_recorder.reset()

        # The number of robots in q_pos_starts might not match the number of environments
        # We set the initial states of the remaining robots far away for visualization purposes
        robot_idxs_keep = self.robot_idxs[: q_pos_starts.shape[0]]
        if self.draw_goal_configuration:
            robot_idxs_keep.append(self.robot_idxs[self.env_goal_configuration_idx])
        robot_idxs_remove = list(set(self.robot_idxs) - set(robot_idxs_keep))

        root_states_aux = self.initial_root_states.clone()
        root_states_aux[robot_idxs_remove, 0] = 50.0
        idxs_remove_int32 = torch.tensor(robot_idxs_remove).to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states_aux),
            gymtorch.unwrap_tensor(idxs_remove_int32),
            len(idxs_remove_int32),
        )
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # Set robot's dof states. Reset to zero velocity
        if q_pos_starts is None:
            # w/o gripper
            q_pos_starts = torch.zeros(
                (self.num_envs, self.robot_num_dofs - self.robot_tr.gripper_q_dim), **self.tensor_args
            )
            q_pos_starts[..., : self.robot_tr.arm_q_dim] = to_torch(
                self.default_dof_pos[: self.robot_tr.arm_q_dim], **self.tensor_args
            )

        assert q_pos_starts.ndim == 2  # Batch, D

        # Make sure joint positions are in the same device as the pipeline
        q_pos_starts = q_pos_starts.to(**self.tensor_args)

        dof_pos_tensor = torch.zeros((self.num_envs, self.robot_num_dofs), **self.tensor_args)
        dof_pos_tensor[: q_pos_starts.shape[0], : self.robot_tr.arm_q_dim] = q_pos_starts
        if self.draw_goal_configuration:
            assert q_pos_goal is not None
            self.goal_joint_position = q_pos_goal
            dof_pos_tensor[self.env_goal_configuration_idx, : self.robot_tr.arm_q_dim] = q_pos_goal

        # Oepn grippers
        if self.robot_tr.gripper_q_dim > 0:
            dof_pos_tensor[..., self.robot_tr.arm_q_dim :] = to_torch(self.robot_dof_upper_limits, **self.tensor_args)[
                None, self.robot_tr.arm_q_dim :
            ]

        self.set_dof_positions(dof_pos_tensor)

        # Get current joint states
        # refresh tensors
        self.gym.refresh_dof_state_tensor(self.sim)
        joint_states_current = self.robot_dof_states

        ###############################################################################################################
        # Clear visualization
        self.gym.clear_lines(self.viewer)
        # reset end effector positions for visualization
        self.end_effector_positions_visualization = [[]] * len(self.robot_handles)

        return joint_states_current

    @property
    def ee_pose_goal(self):
        return self._ee_pose_goal

    @ee_pose_goal.setter
    def ee_pose_goal(self, value):
        self._ee_pose_goal = value

    @ee_pose_goal.getter
    def ee_pose_goal(self):
        return self._ee_pose_goal

    def get_envs(self, first_n_envs=None):
        if self.all_robots_in_one_env:
            envs = self.envs * self.num_envs
        else:
            envs = self.envs
        if first_n_envs is not None:
            envs = envs[:first_n_envs]
        return envs

    def set_dof_positions(self, dof_pos_tensor):
        q_states_des = torch.zeros_like(self.robot_dof_states)
        q_states_des[..., 0] = dof_pos_tensor
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(q_states_des))
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_pos_tensor))

    def step(self, actions, return_viewer_img=False):
        ###############################################################################################################
        # Deploy control based on type
        action_dof_target = torch.zeros_like(self.robot_dof_pos).squeeze(-1)
        action_dof_target[: actions.shape[0], : self.robot_tr.arm_q_dim] = actions[..., : self.robot_tr.arm_q_dim]
        if self.draw_goal_configuration:
            if self.controller_type == "position":
                action_dof_target[self.env_goal_configuration_idx, : self.robot_tr.arm_q_dim] = self.goal_joint_position
            else:
                raise NotImplementedError

        # gripper is open
        if self.robot_tr.gripper_q_dim > 0:
            action_dof_target[..., self.robot_tr.arm_q_dim :] = torch.Tensor(
                [[0.04] * self.robot_tr.gripper_q_dim] * self.num_envs
            )

        ###############################################################################################################
        if self.enable_dynamics:
            # Deploy actions to execute with a controller
            if self.controller_type == "position":
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(action_dof_target))
        else:
            # Sets the simulation to this state and target
            self.set_dof_positions(action_dof_target)

        ###############################################################################################################
        # Step the simulation physics forward
        for _ in range(10):
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        ###############################################################################################################
        # Check collisions between robots and objects
        envs_with_robot_in_contact = self.get_envs_with_contacts(first_n_envs=actions.shape[0])

        ###############################################################################################################
        # Get current joint states
        joint_states_current = self.robot_dof_states
        if self.draw_goal_configuration:
            joint_states_current = joint_states_current[self.env_idxs, ...]

        ###############################################################################################################
        # Visualize and render
        # Clean up
        if not self.draw_end_effector_path:
            self.gym.clear_lines(self.viewer)

        if self.viewer is not None or self.render_camera_global:
            envs = self.get_envs()
            # TODO - implement vectorized versions
            for k, (env, robot_handle) in enumerate(zip(envs, self.robot_handles)):
                # Get end-effector frame
                body_dict = self.gym.get_actor_rigid_body_dict(env, robot_handle)
                props = self.gym.get_actor_rigid_body_states(env, robot_handle, gymapi.STATE_POS)
                ee_pose = props["pose"][:][body_dict[self.robot_end_effector_link_name]]
                ee_position = ee_pose[0]
                ee_transform = gymapi.Transform(p=gymapi.Vec3(*ee_position), r=gymapi.Quat(*ee_pose[1]))

                if self.draw_end_effector_frame:
                    gymutil.draw_lines(self.axes_geom, self.gym, self.viewer, env, ee_transform)

                if self.draw_end_effector_path and self.step_idx % 2 == 0:
                    self.end_effector_positions_visualization[k].append(copy(ee_position))
                    ee_position_l = [ee_position]
                    for ee_position in ee_position_l:
                        link_transform = gymapi.Transform(p=gymapi.Vec3(*ee_position))
                        gymutil.draw_lines(
                            self.sphere_geom_end_effector_l[k], self.gym, self.viewer, env, link_transform
                        )

                # color robots in collision
                if self.collor_robots_in_collision:
                    if k in envs_with_robot_in_contact:
                        n_rigid_bodies = self.gym.get_actor_rigid_body_count(env, robot_handle)
                        for j in range(n_rigid_bodies):
                            self.gym.set_rigid_body_color(
                                env, robot_handle, j, gymapi.MESH_VISUAL_AND_COLLISION, GYM_COLORS["black"]
                            )

                if self.draw_contact_forces:
                    self.gym.draw_env_rigid_contacts(self.viewer, env, gymapi.Vec3(1, 0, 0), 0.5, True)

            if self.draw_ee_pose_goal and self._ee_pose_goal is not None:
                ee_transform = gymapi.Transform(
                    p=gymapi.Vec3(*self._ee_pose_goal[0:3]), r=gymapi.Quat(*self._ee_pose_goal[3:])
                )
                gymutil.draw_lines(self.axes_geom_ee_goal, self.gym, self.viewer, env, ee_transform)
                sphere_geom = gymutil.WireframeSphereGeometry(
                    0.01, 15, 15, gymapi.Transform(), color=(255 / 255.0, 140 / 255.0, 0.0)
                )
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, env, ee_transform)

        # update graphics state
        self.gym.step_graphics(self.sim)

        if self.viewer is not None:
            self.gym.draw_viewer(self.viewer, self.sim, False)
            if self.sync_viewer_with_real_time:
                self.gym.sync_frame_time(self.sim)
            time.sleep(self.viewer_time_between_steps)

        # render viewer camera
        viewer_img = None
        if self.render_camera_global:
            self.gym.render_all_camera_sensors(self.sim)
            viewer_img = self.gym.get_camera_image(
                self.sim, self.middle_env, self.camera_global_handle, gymapi.IMAGE_COLOR
            )
            viewer_img = viewer_img.reshape(viewer_img.shape[0], -1, 4)[..., :3]  # get RGB part
            if self.render_camera_global_append_to_recorder:
                self.camera_global_recorder.append(self.step_idx, viewer_img)

        ###############################################################################################################
        # Update simulation step
        self.step_idx += 1

        if return_viewer_img:
            return joint_states_current, envs_with_robot_in_contact, viewer_img
        else:
            return joint_states_current, envs_with_robot_in_contact

    def get_envs_with_contacts(self, first_n_envs=None):
        # TODO - implement vectorized version
        envs = self.get_envs(first_n_envs=first_n_envs)
        robot_handles = self.robot_handles

        envs_with_robot_in_contact = []
        for env, robot_handle in zip(envs, robot_handles):
            rigid_contacts = self.gym.get_env_rigid_contacts(env)
            if self.all_robots_in_one_env:
                for contact in rigid_contacts:
                    body1_idx = contact[2]
                    env_idx = self.map_rigid_body_idxs_to_env_idx[body1_idx]
                    # discard env_idx if it is not in the selected environments
                    # the environments are selected sequentially
                    if env_idx > len(envs) - 1:
                        continue
                    if env_idx in envs_with_robot_in_contact:
                        pass
                    else:
                        envs_with_robot_in_contact.append(env_idx)
            else:
                if len(rigid_contacts) > 0:
                    env_idx = rigid_contacts[0][0]
                    if env_idx in envs_with_robot_in_contact:
                        pass
                    else:
                        envs_with_robot_in_contact.append(env_idx)
        return envs_with_robot_in_contact

    def take_picture_from_global_camera(self):
        # render viewer camera
        if self.render_camera_global:
            self.gym.render_all_camera_sensors(self.sim)
            viewer_img = self.gym.get_camera_image(
                self.sim, self.middle_env, self.camera_global_handle, gymapi.IMAGE_COLOR
            )
            viewer_img = viewer_img.reshape(viewer_img.shape[0], -1, 4)[..., :3]  # get RGB part
            return viewer_img

    def check_viewer_has_closed(self):
        if self.viewer is None:
            return False
        return self.gym.query_viewer_has_closed(self.viewer)

    def clean_viewer(self):
        self.gym.destroy_viewer(self.viewer)

    def clean_sim(self):
        self.gym.destroy_sim(self.sim)

    def clean_up(self):
        self.clean_viewer()
        self.clean_sim()


class MotionPlanningControllerIsaacGym:

    def __init__(self, motion_planning_isaac_env):
        self.mp_isaac_env = motion_planning_isaac_env

    def execute_trajectories(
        self,
        trajectories,  # shape: (H, B, D)
        q_pos_starts=None,
        q_pos_goal=None,
        n_pre_steps=100,
        n_post_steps=100,
        stop_robot_if_in_contact=False,
        make_video=False,
        **kwargs,
    ):
        assert q_pos_starts is not None
        assert q_pos_goal is not None

        trajectories_copy = trajectories.clone()

        H, B, D = trajectories_copy.shape

        ###############################################################################################################
        # start at the initial position

        joint_states = self.mp_isaac_env.reset(q_pos_starts=q_pos_starts, q_pos_goal=q_pos_goal)

        # first steps -- keep robots in place
        joint_positions_start = joint_states[:, :, 0]  # positions
        for _ in range(n_pre_steps):
            if self.mp_isaac_env.check_viewer_has_closed():
                break
            _, _ = self.mp_isaac_env.step(joint_positions_start)

        # Execute the planned trajectory
        envs_with_robot_in_contact_l = []
        for i, actions in enumerate(trajectories_copy):
            if self.mp_isaac_env.check_viewer_has_closed():
                break
            joint_states, envs_with_robot_in_contact = self.mp_isaac_env.step(actions)
            envs_with_robot_in_contact_l.append(envs_with_robot_in_contact)
            # stop the trajectory if the robots was in contact with the environments
            if stop_robot_if_in_contact and len(envs_with_robot_in_contact) > 0:
                if self.mp_isaac_env.controller_type == "position":
                    trajectories_copy[i:, envs_with_robot_in_contact, :] = actions[envs_with_robot_in_contact, :]

        # last steps -- keep robots in place
        joint_positions_last = joint_states[:, :, 0]  # positions
        for _ in range(n_post_steps):
            if self.mp_isaac_env.check_viewer_has_closed():
                break
            _, _ = self.mp_isaac_env.step(joint_positions_last)

        ###############################################################################################################
        # STATISTICS
        # Get number of trajectories that are in collision
        statistics = DotMap()
        envs_with_robot_in_contact_unique = np.unique(
            [item for sublist in envs_with_robot_in_contact_l for item in sublist]
        )
        statistics.n_trajectories_collision = len(envs_with_robot_in_contact_unique)
        statistics.n_trajectories_free = B - len(envs_with_robot_in_contact_unique)
        statistics.n_trajectories_free_fraction = (B - len(envs_with_robot_in_contact_unique)) / B

        ###############################################################################################################
        # make a video
        if make_video:
            self.mp_isaac_env.camera_global_recorder.make_video(
                n_pre_steps=n_pre_steps, n_post_steps=n_post_steps, **kwargs
            )

        return statistics


if __name__ == "__main__":
    seed = 1
    fix_random_seed(seed)

    device = get_torch_device()
    tensor_args = {"device": "cpu", "dtype": torch.float32}

    # ---------------------------- Environment, Robot, PlanningTask ---------------------------------
    # env = EnvSpheres3D(tensor_args=tensor_args)

    env = EnvWarehouse(rotation_z_axis_deg=-30, tensor_args=tensor_args)

    robot = RobotPanda(tensor_args=tensor_args)

    parametric_trajectory = ParametricTrajectoryBspline(
        n_control_points=18,
        degree=5,
        num_T_pts=128,
        zero_vel_at_start_and_goal=True,
        zero_acc_at_start_and_goal=True,
        remove_outer_control_points=False,
        keep_last_control_point=False,
        trajectory_duration=5.0,
        tensor_args=tensor_args,
    )

    # -------------------------------- Physics --------------------------------
    draw_collision_spheres = False
    robot_asset_file = robot.robot_urdf_collision_spheres_file if draw_collision_spheres else robot.robot_urdf_file
    num_envs = 5
    motion_planning_isaac_env = MotionPlanningIsaacGymEnv(
        env,
        robot,
        asset_root=get_robot_path().as_posix(),
        robot_asset_file=robot_asset_file.replace(get_robot_path().as_posix() + "/", ""),
        controller_type="position",
        num_envs=num_envs,
        all_robots_in_one_env=True,
        show_viewer=True,
        sync_viewer_with_real_time=False,
        viewer_time_between_steps=0.01,
        render_camera_global=True,
        render_camera_global_append_to_recorder=False,
        color_robots=False,
        collor_robots_in_collision=False,
        draw_goal_configuration=False,
        draw_contact_forces=False,
        draw_end_effector_path=False,
        draw_end_effector_frame=False,
        draw_ee_pose_goal=False,
    )

    motion_planning_isaac_env.ee_pose_goal = torch.tensor([0.5, 0, 0.5, 0, 0, 0, 1], **tensor_args)

    motion_planning_controller = MotionPlanningControllerIsaacGym(motion_planning_isaac_env)
    trajectories_joint_pos = (
        torch.rand(robot.q_dim, **tensor_args)
        * (motion_planning_isaac_env.robot_dof_upper_limits[:7] - motion_planning_isaac_env.robot_dof_lower_limits[:7])
        + motion_planning_isaac_env.robot_dof_lower_limits[:7]
    )
    trajectories_joint_pos = trajectories_joint_pos.repeat(100, num_envs, 1)
    trajectories_joint_pos_final = torch.randn_like(trajectories_joint_pos)
    trajectories_joint_pos = trajectories_joint_pos_final - trajectories_joint_pos
    trajectories_joint_pos *= 0.0
    trajectories_joint_pos += 0.2
    trajectories_joint_pos *= 0.0
    motion_planning_controller.execute_trajectories(
        trajectories_joint_pos,
        q_pos_starts=trajectories_joint_pos[0],
        q_pos_goal=trajectories_joint_pos[-1][0],
        n_pre_steps=100,
        n_post_steps=1000,
        make_video=True,
        video_duration=5.0,
        make_gif=False,
    )

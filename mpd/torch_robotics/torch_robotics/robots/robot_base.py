import abc
import atexit
import os
from abc import ABC
from copy import copy, deepcopy
from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET

import einops
import torch
import yaml
from filelock import FileLock
from urdf_parser_py.urdf import URDF, Joint, Link, Visual, Collision, Pose, Sphere, Material, Color

import torchkin
from torch_robotics.robots.torchkin_robot_wrapper import wrapper_torchkin_robot_from_urdf_model
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import (
    q_convert_to_xyzw,
    q_to_euler,
    rotation_matrix_to_q,
)
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
    link_rot_from_link_tensor,
    link_quat_from_link_tensor,
)
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionSelfField
from torch_robotics.torch_utils.torch_utils import to_numpy, to_torch, DEFAULT_TENSOR_ARGS
from torch_robotics.trajectory.utils import finite_difference_vector


def modify_robot_urdf_grasped_object(robot_urdf, robot_urdf_with_spheres, grasped_object):
    parent_link = grasped_object.reference_frame

    # Add the grasped object visual and collision
    link_grasped_object = f"link_{grasped_object.name}"
    joint = Joint(
        name=f"joint_fixed_{grasped_object.name}",
        parent=parent_link,
        child=link_grasped_object,
        joint_type="fixed",
        origin=Pose(
            xyz=to_numpy(grasped_object.pos.squeeze()),
            rpy=to_numpy(q_to_euler(rotation_matrix_to_q(grasped_object.ori)).squeeze()),
        ),
    )
    robot_urdf.add_joint(joint)
    robot_urdf_with_spheres.add_joint(joint)

    geometry_grasped_object = grasped_object.geometry_urdf
    link = Link(
        name=link_grasped_object,
        visual=Visual(geometry_grasped_object),
        # inertial=None,
        collision=Collision(geometry_grasped_object),
        origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
    )
    robot_urdf.add_link(link)
    robot_urdf_with_spheres.add_link(link)

    # Create fixed joints and links for the grasped object collision points
    link_collision_names = []
    for i, point_collision in enumerate(grasped_object.points_for_collision):
        link_collision = f"link_{grasped_object.name}_point_{i}"
        joint = Joint(
            name=f"joint_fixed_{grasped_object.name}_point_{i}",
            parent=link_grasped_object,
            child=link_collision,
            joint_type="fixed",
            origin=Pose(xyz=to_numpy(point_collision)),
        )
        robot_urdf.add_joint(joint)
        robot_urdf_with_spheres.add_joint(deepcopy(joint))

        link = Link(name=link_collision, origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]))
        robot_urdf.add_link(link)

        link = Link(
            name=link_collision,
            # add collision sphere to model
            visual=Visual(
                geometry=Sphere(grasped_object.object_collision_margin),
                material=Material(name="blue", color=Color(0, 0, 1, 1)),
            ),
            origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
        )
        robot_urdf_with_spheres.add_link(link)

        link_collision_names.append(link_collision)

    return robot_urdf, robot_urdf_with_spheres, link_collision_names


def modify_robot_urdf_collision_model(robot_urdf, robot_urdf_with_spheres, collision_spheres_file_path):
    # load collision spheres file
    coll_yml = collision_spheres_file_path
    with open(coll_yml) as file:
        coll_params = yaml.load(file, Loader=yaml.FullLoader)

    # remove self_collision from collision parameters
    coll_params_self_collision = None
    if "self_collision" in coll_params:
        coll_params_self_collision = copy(coll_params["self_collision"])
        del coll_params["self_collision"]

    link_collision_name_dict = dict()
    link_collision_names = []
    link_collision_margins = []
    idx_fk = 0
    link_names = [link.name for link in robot_urdf.links]
    for link_name, spheres_l in coll_params.items():
        for i, sphere in enumerate(spheres_l):
            link_collision = f"{link_name}_{i}"
            joint = Joint(
                name=f"joint_{link_name}_sphere_{i}",
                parent=f"{link_name}",
                child=link_collision,
                joint_type="fixed",
                origin=Pose(xyz=to_numpy(sphere[:3])),
            )
            robot_urdf.add_joint(joint)
            robot_urdf_with_spheres.add_joint(deepcopy(joint))

            link = Link(name=link_collision, origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]))
            robot_urdf.add_link(link)

            link = Link(
                name=link_collision,
                # add collision sphere to model
                visual=Visual(
                    geometry=Sphere(sphere[-1]),
                    # material=Material(name='blue', color=Color(0, 0, 1, 1)),
                    material=Material(name="orange", color=Color(255 / 255, 165 / 255, 0, 1)),
                ),
                origin=Pose(xyz=[0.0, 0.0, 0.0], rpy=[0.0, 0.0, 0.0]),
            )
            robot_urdf_with_spheres.add_link(deepcopy(link))

            link_collision_names.append(link_collision)
            link_collision_margins.append(sphere[-1])

            if link_name in link_collision_name_dict:
                link_collision_name_dict[link_name].append((link_collision, sphere[-1], idx_fk))
            else:
                link_collision_name_dict[link_name] = [(link_collision, sphere[-1], idx_fk)]

            idx_fk += 1

    # create tuples for self collision -- (idx_fk, other_idx_fk, margin, other_margin)
    link_self_collision_tuples = []
    if coll_params_self_collision is not None:
        for link_name in coll_params_self_collision:
            for other_link_name in coll_params_self_collision[link_name]:
                for link_collision, margin, idx_fk in link_collision_name_dict[link_name]:
                    for other_link_collision, other_margin, other_idx_fk in link_collision_name_dict[other_link_name]:
                        link_self_collision_tuples.append((idx_fk, other_idx_fk, margin, other_margin))

    return robot_urdf, robot_urdf_with_spheres, link_collision_names, link_collision_margins, link_self_collision_tuples


def load_joint_limits(joint_limits_file_path):
    with open(joint_limits_file_path) as file:
        joint_limits = yaml.load(file, Loader=yaml.FullLoader)
    return joint_limits


class RobotBase(ABC):
    link_name_ee = None

    def __init__(
        self,
        urdf_robot_file,
        collision_spheres_file_path,
        joint_limits_file_path=None,
        link_name_ee=None,
        gripper_q_dim=0,
        grasped_object=None,
        task_space_dim=3,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):
        self.name = self.__class__.__name__
        assert tensor_args is not None, "tensor_args must be defined"
        self.tensor_args = tensor_args

        assert link_name_ee is not None, "link_name_ee must be defined"
        self.link_name_ee = link_name_ee
        self.gripper_q_dim = gripper_q_dim
        self.grasped_object = grasped_object

        # If the task space is 2D (point mass or plannar robot), then the z coordinate is set to 0
        self.task_space_dim = task_space_dim

        ################################################################################################
        # Get the robot urdf object
        self.robot_urdf_file_raw = urdf_robot_file
        self.robot_urdf = URDF.from_xml_file(urdf_robot_file)
        self.robot_urdf_with_spheres = deepcopy(self.robot_urdf)

        ################################################################################################
        # Robot collision model (links and margins) for object collision avoidance
        self.link_collision_spheres_names = []
        self.link_collision_spheres_radii = []

        # Modify the urdf to append the link and collision points of the grasped object
        self.grasped_object = grasped_object
        if grasped_object is not None:
            (self.robot_urdf, self.robot_urdf_with_spheres, link_collision_names) = modify_robot_urdf_grasped_object(
                self.robot_urdf, self.robot_urdf_with_spheres, grasped_object
            )
            self.link_collision_spheres_names.extend(link_collision_names)
            self.link_collision_spheres_radii.extend(
                [grasped_object.object_collision_margin] * len(link_collision_names)
            )

        # Raw version of the original urdf with the grasped object
        self.robot_urdf_raw = deepcopy(self.robot_urdf)

        # Modify the urdf to append links of the collision model
        (
            self.robot_urdf,
            self.robot_urdf_with_spheres,
            link_collision_names,
            link_collision_margins,
            link_self_collision_tuples,
        ) = modify_robot_urdf_collision_model(
            self.robot_urdf, self.robot_urdf_with_spheres, collision_spheres_file_path
        )
        self.link_collision_spheres_names.extend(link_collision_names)
        self.link_collision_spheres_radii.extend(link_collision_margins)
        assert len(self.link_collision_spheres_names) == len(self.link_collision_spheres_radii)
        self.link_collision_spheres_radii = to_torch(self.link_collision_spheres_radii, **tensor_args)

        # self collision tuples (idx_fk, other_idx_fk, margin, other_margin)
        self.link_self_collision_tuples = link_self_collision_tuples

        # Save the modified urdfs to a file
        self.robot_urdf_file = self.robot_urdf_file_raw.replace(".urdf", f"_tmp_{os.getpid()}.urdf")
        xmlstr = minidom.parseString(ET.tostring(self.robot_urdf.to_xml())).toprettyxml(indent="   ")
        with open(os.path.abspath(self.robot_urdf_file), "w") as f:
            f.write(xmlstr)

        self.robot_urdf_collision_spheres_file = self.robot_urdf_file.replace(".urdf", "_collision_spheres.urdf")
        xmlstr = minidom.parseString(ET.tostring(self.robot_urdf_with_spheres.to_xml())).toprettyxml(indent="   ")
        with open(os.path.abspath(self.robot_urdf_collision_spheres_file), "w") as f:
            f.write(xmlstr)

        atexit.register(self.cleanup)

        ################################################################################################
        # Configuration space limits
        q_limits_lower = []
        q_limits_upper = []
        for joint in self.robot_urdf.joints:
            if joint.joint_type != "fixed":
                q_limits_lower.append(joint.limit.lower)
                q_limits_upper.append(joint.limit.upper)

        self.q_dim = len(q_limits_lower)
        self.gripper_q_dim = gripper_q_dim
        self.arm_q_dim = self.q_dim - self.gripper_q_dim

        self.q_pos_min = to_torch(q_limits_lower, **tensor_args)
        self.q_pos_max = to_torch(q_limits_upper, **tensor_args)
        self.q_pos_min_np = to_numpy(self.q_pos_min)
        self.q_pos_max_np = to_numpy(self.q_pos_max)
        self.q_pos_distribution = torch.distributions.uniform.Uniform(self.q_pos_min, self.q_pos_max)

        ################################################################################################
        # Torchkin robot forward kinematics functions
        # Lock the urdf robot file because of multiprocessing
        self.robot_torchkin = wrapper_torchkin_robot_from_urdf_model(self.robot_urdf, **tensor_args)

        print("-----------------------------------")
        print(f"Torchkin robot: {self.robot_torchkin.name}")
        print(f"Num links: {len(self.robot_torchkin.get_links())}")
        print(f"DOF: {self.robot_torchkin.dof}\n")
        print("-----------------------------------")

        # kinematic functions for object collision (spheres model)
        fk_collision_spheres, jfk_b_collision_spheres, jfk_s_collision_spheres = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=self.link_collision_spheres_names
        )
        self.fk_collision_spheres = fk_collision_spheres
        self.jfk_b_collision_spheres = jfk_b_collision_spheres
        self.jfk_s_collision_spheres = jfk_s_collision_spheres

        # kinematic functions for end-effector link
        fk_ee, jfk_b_ee, jfk_s_ee = torchkin.get_forward_kinematics_fns(
            robot=self.robot_torchkin, link_names=[self.link_name_ee]
        )
        self.fk_ee = fk_ee
        self.jfk_b_ee = jfk_b_ee
        self.jfk_s_ee = jfk_s_ee

        # kinematic functions for all links
        fk_all, jfk_b_all, jfk_s_all = torchkin.get_forward_kinematics_fns(robot=self.robot_torchkin)
        self.fk_all = fk_all
        self.jfk_b_all = jfk_b_all
        self.jfk_s_all = jfk_s_all

        ################################################################################################
        # Self collision field
        self.df_collision_self = None
        if self.link_self_collision_tuples:
            self.df_collision_self = CollisionSelfField(
                robot=self,
                link_self_collision_tuples=self.link_self_collision_tuples,
                tensor_args=tensor_args,
            )

        ################################################################################################
        # Joint limits from configuration file
        self.dq_max = None
        self.dq_max_np = None
        self.ddq_max = None
        self.ddq_max_np = None
        self.joint_limits_file_path = joint_limits_file_path
        if joint_limits_file_path is not None:
            self.joint_limits_d = load_joint_limits(self.joint_limits_file_path)
            self.dq_max = to_torch([v["qdot_max"] for k, v in self.joint_limits_d.items()], **tensor_args)
            self.dq_max_np = to_numpy(self.dq_max)
            self.ddq_max = to_torch([v["qddot_max"] for k, v in self.joint_limits_d.items()], **tensor_args)
            self.ddq_max_np = to_numpy(self.ddq_max)

    def random_q(self, n_samples=10):
        # Random position in configuration space
        q_pos = self.q_pos_distribution.sample((n_samples,))
        return q_pos

    def distance_q(self, q1, q2):
        return torch.linalg.norm(q1 - q2, dim=-1)

    @abc.abstractmethod
    def render(self, ax, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def render_trajectories(self, ax, q_pos_trajs=None, **kwargs):
        raise NotImplementedError

    ################################################################################################
    # Parse state
    def get_position(self, x):
        return x[..., : self.q_dim]

    def get_velocity(self, x):
        return x[..., self.q_dim : 2 * self.q_dim]

    def get_acceleration(self, x):
        return x[..., 2 * self.q_dim : 3 * self.q_dim]

    def get_EE_pose(self, q, flatten_pos_quat=False, quat_xyzw=False):
        _q = q
        if _q.ndim == 1:
            _q = _q.unsqueeze(0)
        if flatten_pos_quat:
            orientation_quat_wxyz = self.get_EE_orientation(_q, rotation_matrix=False)
            orientation_quat = orientation_quat_wxyz
            if quat_xyzw:
                orientation_quat = q_convert_to_xyzw(orientation_quat_wxyz)
            return torch.cat((self.get_EE_position(_q), orientation_quat), dim=-1)
        else:
            pose = self.fk_ee(_q)[0]
            return pose

    def get_EE_position(self, q):
        ee_pose = self.get_EE_pose(q)
        return link_pos_from_link_tensor(ee_pose)

    def get_EE_orientation(self, q, rotation_matrix=True):
        ee_pose = self.get_EE_pose(q)
        if rotation_matrix:
            return link_rot_from_link_tensor(ee_pose)
        else:
            return link_quat_from_link_tensor(ee_pose)

    def fk_map_collision(self, q, **kwargs):
        _q = q
        if _q.ndim == 1:
            _q = _q.unsqueeze(0)  # add batch dimension
        task_space_positions = self.fk_map_collision_impl(_q, **kwargs)
        # Filter the positions from FK to the dimensions of the environment
        # Some environments are defined in 2D, while the robot FK is always defined in 3D
        task_space_positions = task_space_positions[..., : self.task_space_dim]
        return task_space_positions

    def fk_map_collision_impl(self, q, **kwargs):
        # q: (..., q_dim)
        # return: (..., links_collision_positions, 3)
        q_orig_shape = q.shape
        if len(q_orig_shape) == 3:
            b, h, d = q_orig_shape
            q = einops.rearrange(q, "b h d -> (b h) d")
        elif len(q_orig_shape) == 2:
            h = 1
            b, d = q_orig_shape
        else:
            raise NotImplementedError

        link_poses = self.fk_collision_spheres(q)
        links_poses_th = torch.stack(link_poses).transpose(0, 1)

        if len(q_orig_shape) == 3:
            links_poses_th = einops.rearrange(links_poses_th, "(b h) t d1 d2 -> b h t d1 d2", b=b, h=h)

        link_positions_th = link_pos_from_link_tensor(links_poses_th)  # (batch horizon), taskspaces, x_dim

        return link_positions_th

    def cleanup(self):
        if os.path.exists(self.robot_urdf_file):
            os.remove(self.robot_urdf_file)
        if os.path.exists(self.robot_urdf_collision_spheres_file):
            os.remove(self.robot_urdf_collision_spheres_file)

import abc
import datetime
import gc
import math
import os.path
import pickle
import random

import einops
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from mpd.datasets.normalization import DatasetNormalizer
from mpd.datasets.utils import CPU_Unpickler
from pb_ompl.pb_ompl import fit_bspline_to_path
from torch_robotics import robots
from torch_robotics.robots import RobotPointMass2D
from torch_robotics.torch_kinematics_tree.geometrics.utils import (
    link_pos_from_link_tensor,
    link_rot_from_link_tensor,
    rmat_to_flat,
)
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_torch, dict_to_device, to_numpy, DEFAULT_TENSOR_ARGS


def adjust_bspline_number_control_points(
    n_control_points, context_qs, context_ee_goal_pose, zero_vel_at_start_and_goal, zero_acc_at_start_and_goal
):
    # The Unet architecture accepts horizons that are multiples of 2^depth (8 in our case if depth=3).
    # Adjust the control points, such that the number of trainable control points is a multiple of 8.
    removed_control_points = 0
    trainable_control_points = n_control_points
    if context_qs and context_ee_goal_pose:
        trainable_control_points -= 1
        removed_control_points += 1
    elif context_qs:
        trainable_control_points -= 2
        removed_control_points += 2

    if zero_vel_at_start_and_goal:
        trainable_control_points -= 2
        removed_control_points += 2
    if zero_acc_at_start_and_goal:
        trainable_control_points -= 2
        removed_control_points += 2

    # make sure the number of trainable control points is a multiple of 8 (next closest multiple of 8)
    trainable_control_points = int(math.ceil(trainable_control_points / 8) * 8)

    return trainable_control_points + removed_control_points, removed_control_points


class TrajectoryDatasetBspline(Dataset, abc.ABC):

    def __init__(
        self,
        planning_task=None,
        base_dir=None,
        dataset_file_merged="dataset_merged.hdf5",
        context_qs=True,
        context_ee_goal_pose=False,
        normalizer="SafeLimitsNormalizer",
        normalize_ee_pose_goal=True,
        reload_data=False,
        preload_data_to_device=False,
        n_task_samples=-1,
        tensor_args=DEFAULT_TENSOR_ARGS,
        **kwargs,
    ):

        self.planning_task = planning_task

        self.base_dir = base_dir
        self.dataset_file_merged = dataset_file_merged

        self.tensor_args = tensor_args

        ######################################################################################
        # -------------------------------- Context variables ---------------------------------
        # Use joint start (and goal) as a context variable
        self.context_qs = context_qs

        # Use end-effector pose goal as a context variable
        self.context_ee_goal_pose = context_ee_goal_pose

        ######################################################################################
        # -------------------------------- Load trajectories ---------------------------------
        # Dataset fields
        self.field_key_control_points = "control_points"
        self.field_key_q_start = "q_start"
        self.field_key_q_goal = "q_goal"
        self.field_key_context_qs = "qs"
        self.field_key_context_ee_goal_pose = "ee_goal_pose"
        self.field_key_context_ee_goal_orientation = "ee_goal_orientation"
        self.field_key_context_ee_goal_position = "ee_goal_position"
        self.fields = {}

        # ------------ load data ------------
        self.map_task_id_to_control_points_id = {}
        self.map_control_points_id_to_task_id = {}
        self.reload_data = reload_data
        self.preload_data_to_device = preload_data_to_device
        self.load_data(n_task_samples=n_task_samples)
        # possibly move the dataset to the gpu
        if self.preload_data_to_device:
            self.fields = dict_to_device(self.fields, **self.tensor_args)
        torch.cuda.empty_cache()
        gc.collect()

        # dimensions
        b, h, d = self.dataset_shape = self.fields[self.field_key_control_points].shape
        self.n_trajectories = b
        self.n_learnable_control_points = h
        self.state_dim = d  # state dimension used for the generative model
        self.control_points_dim = (self.n_learnable_control_points, d)

        self.context_q_dim = self.fields[self.field_key_context_qs].shape[-1]

        # -------------------------------- Normalize data ---------------------------------
        # normalize the data
        self.normalizer = DatasetNormalizer(self.fields, normalizer=normalizer)
        self.normalizer_keys = [
            self.field_key_control_points,
            self.field_key_context_qs,
            self.field_key_context_ee_goal_orientation,
            self.field_key_context_ee_goal_position,
        ]
        self.normalize_ee_pose_goal = normalize_ee_pose_goal
        if normalize_ee_pose_goal:
            # do not normalize the orientation, only the position part of the end-effector pose
            self.normalizer_keys.append(self.field_key_context_ee_goal_position)

        self.normalize_all_data(*self.normalizer_keys)

        # Do not normalize rotation matrices
        # for compatilibity, set the normalized orientation to the unnormalized one
        self.fields[self.field_key_context_ee_goal_orientation + "_normalized"] = self.fields[
            self.field_key_context_ee_goal_orientation
        ]

        if not normalize_ee_pose_goal:
            # normalized data is the same as the original data
            self.fields[self.field_key_context_ee_goal_pose + "_normalized"] = self.fields[
                self.field_key_context_ee_goal_pose
            ]

        # move normalize class members data to devices
        self.normalizer.move_to_device(device=self.tensor_args["device"])

    def reload_data_fn(self, data_reload_pickle_path, n_task_samples=-1):
        with open(data_reload_pickle_path, "rb") as fp:
            data = CPU_Unpickler(fp).load()
        self.fields = data["fields"]
        self.map_task_id_to_control_points_id = data["map_task_id_to_control_points_id"]

        if n_task_samples != -1:
            n_task_samples = min(n_task_samples, len(self.map_task_id_to_control_points_id))
            dict_tmp = {
                key: self.map_task_id_to_control_points_id[key]
                for key in np.random.choice(
                    list(self.map_task_id_to_control_points_id.keys()), n_task_samples, replace=False
                )
            }
            self.map_task_id_to_control_points_id = dict_tmp

        self.map_control_points_id_to_task_id = data["map_control_points_id_to_task_id"]

    def load_data(self, n_task_samples=-1):
        # load data into CPU RAM
        with TimerCUDA() as t_load_data:
            print(f"Loading data ...")

            # File name for data reload
            data_reload_prefix = f'{self.dataset_file_merged.replace(".hdf5", "")}_reload'
            data_reload_prefix += f"-ntasks_{n_task_samples}"
            data_reload_prefix += f"--bspline"
            data_reload_prefix += f"-degree_{self.planning_task.parametric_trajectory.bspline.d}"
            data_reload_prefix += f"-n_pts_{self.planning_task.parametric_trajectory.bspline.n_pts}"
            data_reload_prefix += f"-zero_vel_{self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal}"
            data_reload_prefix += f"-zero_acc_{self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal}"
            data_reload_file_path = os.path.join(self.base_dir, f"{data_reload_prefix}.pickle")

            if os.path.exists(data_reload_file_path) and not self.reload_data:
                # load the pre-processed dataset
                self.reload_data_fn(data_reload_file_path, n_task_samples=n_task_samples)
            else:
                # load dataset file
                dataset_h5 = h5py.File(os.path.join(self.base_dir, self.dataset_file_merged), "r")

                # load trajectories
                inner_control_points_all = []
                q_start_all = []
                q_goal_all = []

                task_ids_processed = []
                # fit a bspline to each path
                cps_idx = 0
                n_discarded_trajectories = 0
                for i, path in enumerate(dataset_h5["sol_path"]):
                    # check the number of processed tasks
                    if n_task_samples != -1 and i > 0:
                        task_ids_processed.append(dataset_h5["task_id"][i])
                        task_ids_processed = list(set(task_ids_processed))
                        if len(task_ids_processed) >= n_task_samples:
                            break

                    # fit a bspline or load from data
                    if "bspline_params_cc" in dataset_h5:
                        bspline_params = (
                            dataset_h5["bspline_params_tt"][i],
                            dataset_h5["bspline_params_cc"][i],
                            dataset_h5["bspline_params_k"][i],
                        )
                    else:
                        # fit a spline to the path
                        try:
                            bspline_params = fit_bspline_to_path(
                                dataset_h5["sol_path"][i],
                                bspline_degree=self.planning_task.parametric_trajectory.bspline.d,
                                bspline_num_control_points=self.planning_task.parametric_trajectory.bspline.n_pts,
                                bspline_zero_vel_at_start_and_goal=self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal,
                                bspline_zero_acc_at_start_and_goal=self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal,
                                debug=False,
                            )
                            _, cc_tmp, _ = bspline_params
                            # discard the trajectory if the bspline coefficients are too large wrt the joint limits
                            if np.any(cc_tmp.min(1) <= 2 * to_numpy(self.planning_task.robot.q_pos_min)) or np.any(
                                cc_tmp.max(1) >= 2 * to_numpy(self.planning_task.robot.q_pos_max)
                            ):
                                n_discarded_trajectories += 1
                                raise Exception
                        except:
                            continue

                    tt, cc, k = bspline_params

                    # bspline coefficients (control points)
                    cc_np = np.array(cc)
                    if isinstance(self.planning_task.robot, robots.RobotPanda):
                        # TODO - fix for other manipulators
                        # If the number of joints is greater than 7, remove the last two points, which can be from
                        # the gripper.
                        if cc_np.shape[0] > 9:
                            cc_np = cc_np[:7, :]

                    # load to the cpu to speed up loading and save gpu memory
                    control_points = to_torch(cc_np, dtype=self.tensor_args["dtype"], device="cpu").transpose(0, 1)

                    # start and goal joint positions are the first and last control points by definition
                    q_start_all.append(control_points[0])
                    q_goal_all.append(control_points[-1])

                    # If joint start and goal are used as context variables, remove the first and last control points
                    # from the learned control points.
                    # Same for zero velocity and acceleration at start and goal.
                    inner_control_points = self.planning_task.parametric_trajectory.remove_control_points_fn(
                        control_points
                    )

                    inner_control_points_all.append(inner_control_points)

                    # map control points to task ids
                    task_id = dataset_h5["task_id"][i]
                    self.map_control_points_id_to_task_id[cps_idx] = task_id
                    # map task ids to control points
                    if task_id in self.map_task_id_to_control_points_id:
                        self.map_task_id_to_control_points_id[task_id].append(cps_idx)
                    else:
                        self.map_task_id_to_control_points_id[task_id] = [cps_idx]

                    cps_idx += 1

                    if i % 20000 == 0 or i == len(dataset_h5["sol_path"]) - 1:
                        print(
                            f"Time spent: {str(datetime.timedelta(seconds=t_load_data.elapsed))} - "
                            f'loaded {i}/{len(dataset_h5["sol_path"])} '
                            f'({i/len(dataset_h5["sol_path"]):.2%}) trajectories.'
                        )

                print(f'Number of discarded trajectories: {n_discarded_trajectories}/{len(dataset_h5["sol_path"])}')

                # learnable inner control points
                inner_control_points_tensor = torch.stack(inner_control_points_all)
                self.fields[self.field_key_control_points] = inner_control_points_tensor

                # update fields for all samples
                self.fields = self.build_fields_data_sample(
                    self.fields, torch.stack(q_start_all), torch.stack(q_goal_all), device="cpu"
                )

                # Compute collision free statistics of the fitted bspline (this can take a while).
                percentage_free_trajs_l, percentage_collision_intensity_l = self.run_collision_statistics()

                # Save data to disk to speed up loading the next time.
                data_to_save = {
                    "fields": self.fields,
                    "map_task_id_to_control_points_id": self.map_task_id_to_control_points_id,
                    "map_control_points_id_to_task_id": self.map_control_points_id_to_task_id,
                    "percentage_free_trajs": (np.mean(percentage_free_trajs_l), np.std(percentage_free_trajs_l)),
                    "percentage_collision_intensity": (
                        np.mean(percentage_collision_intensity_l),
                        np.std(percentage_collision_intensity_l),
                    ),
                }
                pickle.dump(data_to_save, open(data_reload_file_path, "wb"))

            print("... done loading data.")
            print(f"Loading data took {t_load_data.elapsed:.2f} seconds.")

    def build_fields_data_sample(self, fields_d, q_start, q_goal, ee_pose_goal=None, device=None, **kwargs):
        fields_d[self.field_key_q_start] = q_start
        fields_d[self.field_key_q_goal] = q_goal

        # task: start and goal control_points [n_trajectories, (state_dim or 2 * state_dim)]
        if self.context_qs and self.context_ee_goal_pose:
            context_qs = q_start
        else:
            context_qs = torch.cat((q_start, q_goal), dim=-1)
        fields_d[self.field_key_context_qs] = context_qs

        # Context data
        # end-effector pose goal
        if ee_pose_goal is None:
            ee_pose_goal = self.planning_task.robot.get_EE_pose(
                to_torch(q_goal, **self.planning_task.robot.tensor_args)
            ).to(device if device is not None else self.tensor_args["device"])
        ee_pose_goal_orientation = rmat_to_flat(link_rot_from_link_tensor(ee_pose_goal))
        ee_pose_goal_position = link_pos_from_link_tensor(ee_pose_goal)
        fields_d[self.field_key_context_ee_goal_pose] = ee_pose_goal
        fields_d[self.field_key_context_ee_goal_orientation] = ee_pose_goal_orientation
        fields_d[self.field_key_context_ee_goal_position] = ee_pose_goal_position

        return fields_d

    def normalize_all_data(self, *keys):
        for key in keys:
            self.fields[f"{key}_normalized"] = self.normalizer(self.fields[f"{key}"], key)

    def __repr__(self):
        msg = (
            f"{self.__class__.__name__}\n"
            f"n_tasks: {len(self.map_task_id_to_control_points_id)}\n"
            f"n_trajs: {self.n_trajectories}\n"
            f"control_points_dim: {self.control_points_dim}\n"
        )
        return msg

    def __len__(self):
        return self.n_trajectories

    def __getitem__(self, index):
        # Generates one sample of data - one trajectory and tasks
        data = {}
        for field in self.fields:
            data[field] = self.fields[field][index]

        # build hard conditions
        hard_conds = self.get_hard_conditions(data)
        data["hard_conds"] = hard_conds

        return data

    def create_data_sample_normalized(self, q_start_pos, q_goal_pos, ee_pose_goal=None, **kwargs):
        # create a data sample
        data_sample = {}
        data_sample = self.build_fields_data_sample(data_sample, q_start_pos, q_goal_pos, ee_pose_goal=ee_pose_goal)

        for k in list(data_sample.keys()):
            v = data_sample[k]
            # do not normalize the orientation, only the position part of the end-effector goal pose
            if k == self.field_key_context_ee_goal_orientation:
                data_sample[f"{k}_normalized"] = data_sample[k]
            elif k == self.field_key_context_ee_goal_pose:
                pass
            else:
                data_sample[f"{k}_normalized"] = self.normalize(v, k)

        if not self.preload_data_to_device:
            data_sample = dict_to_device(data_sample, **self.tensor_args)

        # build hard conditions
        hard_conds = self.get_hard_conditions(data_sample)
        data_sample["hard_conds"] = hard_conds

        return data_sample

    def get_hard_conditions(self, data_d, **kwargs):
        hard_conds = {}
        if not self.context_qs:
            # start and goal joint positions
            cps_normalized = data_d[f"{self.field_key_control_points}_normalized"]
            cps_start_normalized = cps_normalized[0]
            cps_goal_normalized = cps_normalized[-1]

            horizon = self.n_learnable_control_points

            # The Initial and final velocity and acceleration should be zero
            # Set the second and second-to-last control points to be the same as the first and last control points
            # Set the third and third-to-last control points to be the same as the first and last control points
            # Do not set a hard condition on the last control point joint position if the end-effector goal is used as
            # a context variable
            hard_conds = {0: cps_start_normalized}
            if self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal:
                hard_conds[1] = cps_start_normalized
            if self.planning_task.parametric_trajectory.zero_acc_at_start_and_goal:
                hard_conds[2] = cps_start_normalized

            if not self.context_ee_goal_pose:
                hard_conds[horizon - 1] = cps_goal_normalized
                if self.planning_task.parametric_trajectory.zero_vel_at_start_and_goal:
                    hard_conds[horizon - 2] = cps_goal_normalized
                if self.planning_task.plannerlf.zero_acc_at_start_and_goal:
                    hard_conds[horizon - 3] = cps_goal_normalized

        return hard_conds

    def build_context(self, data_sample):
        context_d = {}
        if self.context_qs and self.context_ee_goal_pose:
            context_d = {
                self.field_key_context_qs: data_sample[self.field_key_context_qs],
                f"{self.field_key_context_qs}_normalized": data_sample[f"{self.field_key_context_qs}_normalized"],
                self.field_key_context_ee_goal_pose: data_sample[self.field_key_context_ee_goal_pose],
                self.field_key_context_ee_goal_orientation: data_sample[self.field_key_context_ee_goal_orientation],
                f"{self.field_key_context_ee_goal_orientation}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_orientation}_normalized"
                ],
                self.field_key_context_ee_goal_position: data_sample[self.field_key_context_ee_goal_position],
                f"{self.field_key_context_ee_goal_position}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_position}_normalized"
                ],
            }
        elif self.context_qs:
            context_d = {
                self.field_key_context_qs: data_sample[self.field_key_context_qs],
                f"{self.field_key_context_qs}_normalized": data_sample[f"{self.field_key_context_qs}_normalized"],
            }
        elif self.context_ee_goal_pose:
            context_d = {
                self.field_key_context_ee_goal_pose: data_sample[self.field_key_context_ee_goal_pose],
                self.field_key_context_ee_goal_orientation: data_sample[self.field_key_context_ee_goal_orientation],
                f"{self.field_key_context_ee_goal_orientation}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_orientation}_normalized"
                ],
                self.field_key_context_ee_goal_position: data_sample[self.field_key_context_ee_goal_position],
                f"{self.field_key_context_ee_goal_position}_normalized": data_sample[
                    f"{self.field_key_context_ee_goal_position}_normalized"
                ],
            }
        return context_d

    def get_unnormalized(self, index):
        raise NotImplementedError

    def unnormalize(self, x, key):
        return self.normalizer.unnormalize(x, key)

    def normalize(self, x, key):
        return self.normalizer.normalize(x, key)

    def unnormalize_control_points(self, x):
        return self.unnormalize(x, self.field_key_control_points)

    def grad_unnormalized_wrt_control_points_normalized(self, x):
        return self.normalizer.grad_x_wrt_x_normalized(x, self.field_key_control_points)

    def normalize_control_points(self, x):
        return self.normalize(x, self.field_key_control_points)

    def unnormalize_tasks(self, x):
        return self.unnormalize(x, self.field_key_context_qs)

    def normalize_tasks(self, x):
        return self.normalize(x, self.field_key_context_qs)

    def normalize_ee_pose(self, x):
        if not self.normalize_ee_pose_goal:
            return x
        # normalize only the position part of the end-effector pose
        position = link_pos_from_link_tensor(x)
        position_normalized = self.normalize(position, self.field_key_context_ee_goal_position)
        x[..., :3, 3] = position_normalized
        return x

    def render(
        self,
        task_id=None,
        render_joint_trajectories=False,
        render_robot_trajectories=False,
        render_n_robot_trajectories=-1,
        **kwargs,
    ):
        # -------------------------------- Visualize ---------------------------------
        if task_id is None:
            task_id = np.random.choice(list(self.map_task_id_to_control_points_id.keys()))
        idxs = np.asarray(self.map_task_id_to_control_points_id[task_id])
        control_points = self.fields[self.field_key_control_points][idxs]
        control_points = to_torch(control_points, **self.tensor_args)

        # Get trajectories from parametric trajectory
        q_pos_start = to_torch(self.fields[self.field_key_q_start][idxs], **self.tensor_args)
        q_pos_goal = to_torch(self.fields[self.field_key_q_goal][idxs], **self.tensor_args)
        q_pos_trajs = self.planning_task.parametric_trajectory.get_q_trajectory(
            control_points, q_pos_start, q_pos_goal, get_type=["pos"]
        )["pos"]

        fig1, axs1, fig2, axs2 = [None] * 4
        if render_joint_trajectories:
            fig1, axs1 = self.planning_task.plot_joint_space_trajectories(
                q_pos_trajs=q_pos_trajs,
                q_vel_trajs=None,
                q_acc_trajs=None,
                q_pos_start=q_pos_start[0],
                q_pos_goal=q_pos_goal[0],
                # vel_start_state=torch.zeros_like(q_start), vel_goal_state=torch.zeros_like(q_goal),
                set_q_pos_limits=True,
                set_q_vel_limits=True,
                set_q_acc_limits=True,
            )

        if render_robot_trajectories:
            # render n random trajectories
            # TODO - implement parallel rendering
            if render_n_robot_trajectories != -1:
                idxs_n = np.random.choice(q_pos_trajs.shape[0], render_n_robot_trajectories)
            else:
                idxs_n = np.arange(q_pos_trajs.shape[0])
            q_pos_trajs = q_pos_trajs[idxs_n]
            ee_goal_position = self.fields[self.field_key_context_ee_goal_position][idxs[idxs_n]]
            fig2, axs2 = self.planning_task.render_robot_trajectories(
                q_pos_trajs=q_pos_trajs,
                q_pos_start=q_pos_start[0],
                q_pos_goal=q_pos_goal[0],
                ee_goal_position=ee_goal_position[0],
                **kwargs,
            )
            if isinstance(self.planning_task.robot, RobotPointMass2D):
                for cps in to_numpy(control_points[idxs_n]):
                    axs2.scatter(cps[:, 0], cps[:, 1], c="b", s=10, zorder=100)

        return fig1, axs1, fig2, axs2

    def run_collision_statistics(self, chunk_size=2000):
        # checks the collision statistics of the fitted bsplines

        # get all control points idxs
        idxs_cps_sequential = np.arange(self.fields[self.field_key_control_points].shape[0])

        # loop idxs in chunks due to memory constraints
        percentage_free_trajs_l = []
        percentage_collision_intensity_l = []
        for idxs_chunk in tqdm(
            chunker(idxs_cps_sequential, chunk_size), total=np.ceil(len(idxs_cps_sequential) / chunk_size)
        ):
            control_points = self.fields[self.field_key_control_points][idxs_chunk]
            control_points = to_torch(control_points, **self.tensor_args)

            # Get trajectories
            q_start = self.fields[self.field_key_q_start][idxs_chunk]
            q_start = to_torch(q_start, **self.tensor_args)
            q_goal = self.fields[self.field_key_q_goal][idxs_chunk]
            q_goal = to_torch(q_goal, **self.tensor_args)
            q_trajs_pos = self.planning_task.parametric_trajectory.get_q_trajectory(
                control_points, q_start, q_goal, get_type=("pos",)
            )["pos"]

            percentage_free_trajs_l.append(self.planning_task.compute_fraction_valid_trajs(q_trajs_pos))
            percentage_collision_intensity_l.append(
                self.planning_task.compute_collision_intensity_trajs(q_trajs_pos).cpu().item()
            )

        print("---------------------------------")
        print("Dataset statistics after fitting")
        print(f"percentage free trajs: ({np.mean(percentage_free_trajs_l):.4f}, {np.std(percentage_free_trajs_l):.4f})")
        print(
            f"percentage collision intensity: ({np.mean(percentage_collision_intensity_l):.4f}, {np.std(percentage_collision_intensity_l):.4f})"
        )
        print("---------------------------------\n")

        return percentage_free_trajs_l, percentage_collision_intensity_l

    def get_knn_control_points(self, q_start, q_goal, EE_pose_goal, k=2, normalized=True, **kwargs):
        assert self.context_qs, (
            "The joint positions should be used as context variables. " "TODO - implement for no context variables."
        )
        # get the k nearest neighbors context variables
        # task: start and goal control_points [n_trajectories, (state_dim or 2 * state_dim)]
        if self.context_qs and self.context_ee_goal_pose:
            # The context dataset is the concatenation of the start joint positions,
            # the end-effector goal orientation (flattened)
            # and the end-effector goal position
            # This is context has the same format as the input to the generative model
            context_dataset = torch.cat(
                [
                    self.fields[self.field_key_q_start],
                    self.fields[self.field_key_context_ee_goal_orientation],  # orientation flattened
                    self.fields[self.field_key_context_ee_goal_position],
                ],
                dim=-1,
            )
            context_dataset = to_torch(context_dataset, device=q_start.device)
            context_query = torch.cat(
                [
                    q_start,
                    einops.rearrange(EE_pose_goal[..., :3, :3], "... d e -> ... (d e)"),
                    EE_pose_goal[..., :3, 3],
                ],
                dim=-1,
            )[None, ...]
        elif self.context_qs:
            context_dataset = to_torch(self.fields[self.field_key_context_qs], device=q_start.device)
            context_query = torch.cat((q_start, q_goal), dim=-1)[None, ...]
        else:
            raise NotImplementedError

        # Compute the Euclidean distance between the query and the dataset
        dist = torch.linalg.vector_norm(context_dataset - context_query, dim=-1)
        # We evaluate our models using the training and validation sets of the dataset.
        # Therefore, it happens that for a context query value, the nearest neighbor is the query itself.
        # In a real scenario, this would not happen.
        # To prevent this, we discard samples that have a distance equal to 0.
        # We augment the number of neighbors > k + n, where n is the number of trajectories per task.
        _, cps_idxs = random.choice(list(self.map_task_id_to_control_points_id.items()))
        # Compute the neighbors and discard the ones with zero distance
        knn = dist.topk(k + len(cps_idxs) + 100, largest=False)
        idxs = knn.indices[knn.values > 0][:k]

        if normalized:
            return self.fields[f"{self.field_key_control_points}_normalized"].to(**self.tensor_args)[idxs]
        else:
            return self.fields[f"{self.field_key_control_points}"].to(**self.tensor_args)[idxs]


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))

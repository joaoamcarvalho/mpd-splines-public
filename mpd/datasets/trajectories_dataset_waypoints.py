import datetime
import math
import os.path
import pickle

import h5py
import numpy as np
import torch

from mpd.datasets.trajectories_dataset_bspline import TrajectoryDatasetBspline
from torch_robotics import robots
from torch_robotics.trajectory.utils import interpolate_points_v1
from torch_robotics.torch_utils.torch_timer import TimerCUDA


def adjust_waypoints(n_waypoints, context_qs, context_ee_goal_pose):
    # The Unet architecture accepts horizons that are multiples of 2^depth (8 in our case).
    # Adjust the waypoints, such that the number of trainable waypoints is a multiple of 8.
    removed_waypoints = 0
    trainable_waypoints = n_waypoints
    if context_qs and context_ee_goal_pose:
        trainable_waypoints -= 1
        removed_waypoints += 1
    elif context_qs:
        trainable_waypoints -= 2
        removed_waypoints += 2

    # make sure the number of trainable waypoints is a multiple of 8 (next closest multiple of 8)
    trainable_waypoints = int(math.ceil(trainable_waypoints / 8) * 8)

    return trainable_waypoints + removed_waypoints, removed_waypoints


class TrajectoryDatasetWaypoints(TrajectoryDatasetBspline):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data(self, n_task_samples=-1):
        # load data into CPU RAM
        with TimerCUDA() as t_load_data:
            print(f"Loading data ...")

            # Data reload file name
            data_reload_prefix = f'{self.dataset_file_merged.replace(".hdf5", "")}_reload'
            data_reload_prefix += f"-ntasks_{n_task_samples}"
            data_reload_prefix += f"--waypoints"
            data_reload_prefix += f"-n_pts_{self.planning_task.parametric_trajectory.n_control_points}"
            data_reload_pickle_path = os.path.join(self.base_dir, f"{data_reload_prefix}.pickle")

            if os.path.exists(data_reload_pickle_path) and not self.reload_data:
                # load the pre-processed dataset
                self.reload_data_fn(data_reload_pickle_path, n_task_samples=n_task_samples)
            else:
                # load the dataset to the cpu first
                # load dataset file
                dataset_h5 = h5py.File(os.path.join(self.base_dir, self.dataset_file_merged), "r")

                # load trajectories
                inner_control_points_all = []
                q_start_all = []
                q_goal_all = []

                task_ids_processed = []
                # linearly interpolate each trajectory to the desired number of control points
                cps_idx = 0
                for i, path in enumerate(dataset_h5["sol_path"]):
                    # check the number of processed tasks
                    if n_task_samples != -1 and i > 0:
                        task_ids_processed.append(dataset_h5["task_id"][i])
                        task_ids_processed = list(set(task_ids_processed))
                        if len(task_ids_processed) >= n_task_samples:
                            break

                    sol_path = dataset_h5["sol_path"][i]
                    control_points = interpolate_points_v1(
                        torch.from_numpy(sol_path).to(dtype=self.tensor_args["dtype"])[None, ...],
                        self.planning_task.parametric_trajectory.n_control_points,
                    ).squeeze()

                    if isinstance(self.planning_task.robot, robots.RobotPanda):
                        # If the number of joints is 9, then remove the last two points, which can be
                        # the two gripper fingers.
                        if control_points.shape[-1] == 9:
                            control_points = control_points[..., :7]

                    # start and goal joint positions are the first and last control points by definition of the bspline
                    q_start_all.append(control_points[0])
                    q_goal_all.append(control_points[-1])

                    # If joint start and goal are used as context variables,
                    # remove the first and last control points from the dataset.
                    # Do the same for zero velocity and acceleration at start and goal.
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

                # waypoints
                inner_control_points_tensor = torch.stack(inner_control_points_all)
                self.fields[self.field_key_control_points] = inner_control_points_tensor

                # update fields for all samples
                self.fields = self.build_fields_data_sample(
                    self.fields,
                    torch.stack(q_start_all),  # start and goal joint positions
                    torch.stack(q_goal_all),
                    device="cpu",
                )

                # compute collision free statistics of the fitted bspline
                percentage_free_trajs_l, percentage_collision_intensity_l = self.run_collision_statistics()

                # save data to disk to speed up loading the next time
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
                pickle.dump(data_to_save, open(data_reload_pickle_path, "wb"))

            print("... done loading data.")
            print(f"Loading data took {t_load_data.elapsed:.2f} seconds.")

    def get_hard_conditions(self, data_d, **kwargs):
        hard_conds = {}
        if not self.context_qs:
            # start and goal joint positions
            control_points_normalized = data_d[f"{self.field_key_control_points}_normalized"]
            control_points_start_normalized = control_points_normalized[0]
            control_points_goal_normalized = control_points_normalized[-1]

            horizon = self.n_learnable_control_points

            # Do not set a hard condition on the last control point joint position if the end-effector goal is used as
            # a context variable
            hard_conds = {0: control_points_start_normalized}

            if not self.context_ee_goal_pose:
                hard_conds[horizon - 1] = control_points_goal_normalized

        return hard_conds

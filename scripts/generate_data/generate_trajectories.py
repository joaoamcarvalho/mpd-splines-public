import os
import pathlib
import pickle
import random
import time
from copy import deepcopy, copy
from xml.dom import minidom
from xml.etree import ElementTree as ET

import h5py
import numpy as np
import pybullet as p
import torch
import yaml
from joblib import Parallel, delayed
from pybullet_utils import bullet_client
from tqdm import tqdm

from experiment_launcher import single_experiment_yaml, run_experiment
from experiment_launcher.utils import fix_random_seed
from mpd.paths import DATA_GENERATION_CFGS_PATH
from pb_ompl.pb_ompl import PbOMPLRobot, PbOMPL, add_sphere, add_box
from torch_robotics import environments, robots
from torch_robotics.environments.primitives import MultiSphereField, MultiBoxField
from torch_robotics.torch_kinematics_tree.geometrics.quaternion import q_convert_to_xyzw, rotation_matrix_to_q
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS

from scipy.spatial.transform import Rotation


class GenerateDataOMPL:
    def __init__(
        self,
        env_id,
        robot_id,
        min_distance_robot_env=0.02,
        planner="RRTConnect",
        tensor_args=DEFAULT_TENSOR_ARGS,
        pybullet_mode="DIRECT",
        debug=False,
        env_tr=None,
        robot_tr=None,
        **kwargs,
    ):
        self.tensor_args = tensor_args

        # -------------------------------- Load env, robot, task ---------------------------------
        # Environment
        if env_tr is not None:
            self.env_tr = deepcopy(env_tr)
        else:
            env_class = getattr(environments, env_id)
            self.env_tr = env_class(
                precompute_sdf_obj_fixed=False, precompute_sdf_obj_extra=False, tensor_args=tensor_args
            )

        # Robot from torch_robotics
        if robot_tr is not None:
            self.robot_tr = deepcopy(robot_tr)
        else:
            robot_class_tr = getattr(robots, robot_id)
            self.robot_tr = robot_class_tr(tensor_args=tensor_args, **kwargs)

        # --------------------------------------------------------------------------------------------
        # Setup pybullet_ompl
        self.obstacles = []

        # setup pybullet client due to multi-threading
        self.pybullet_client = bullet_client.BulletClient(
            connection_mode=p.GUI if pybullet_mode == "GUI" else p.DIRECT, options=""
        )
        self.pybullet_client.setGravity(0, 0, 0.0)  # no gravity
        self.pybullet_client.setTimeStep(1.0 / 240.0)

        # For compability, create a temporary file to store the robot urdf
        path = pathlib.Path(self.robot_tr.robot_urdf_file)
        robot_urdf_file_tmp = path.with_name(path.stem + f"-{os.getpid()}" + path.suffix).as_posix()
        robot_urdf_xmlstr = minidom.parseString(ET.tostring(self.robot_tr.robot_urdf_raw.to_xml())).toprettyxml(
            indent="   "
        )
        with open(robot_urdf_file_tmp, "w") as f:
            f.write(robot_urdf_xmlstr)

        robot_id_bullet = self.pybullet_client.loadURDF(
            robot_urdf_file_tmp,
            (0, 0, 0),
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION,
        )
        robot = PbOMPLRobot(
            self.pybullet_client,
            robot_id_bullet,
            urdf_path=robot_urdf_file_tmp,
            link_name_ee=self.robot_tr.link_name_ee,
        )
        self.robot_pbompl = robot
        os.remove(robot_urdf_file_tmp)  # delete the temporary file

        # setup pb_ompl
        self.pbompl_interface = PbOMPL(
            self.pybullet_client, self.robot_pbompl, self.obstacles, min_distance_robot_env=min_distance_robot_env
        )
        self.pbompl_interface.set_planner(planner)

        # add obstacles
        self.add_obstacles()

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            self.pybullet_client.removeBody(obstacle)
        self.obstacles = []

    def add_obstacles(self, default_height_2d=0.05):
        for obj_list, color in zip(
            [self.env_tr.get_obj_fixed_list(), self.env_tr.get_obj_extra_list()],
            [(220.0 / 255.0, 220.0 / 255.0, 220.0 / 255.0, 1.0), (1, 0, 0, 1)],
        ):
            for obj in obj_list:
                obj_position, obj_orientation = obj.get_position_orientation()
                obj_orientation = q_convert_to_xyzw(rotation_matrix_to_q(obj_orientation))
                for single_primitive in obj.get_all_single_primitives():
                    if isinstance(single_primitive, MultiSphereField):
                        center, radius = single_primitive.centers, single_primitive.radii
                        if len(center) == 2:
                            center = torch.cat([center, torch.zeros(1, **self.tensor_args)])
                        center += obj_position
                        self.obstacles.append(
                            add_sphere(
                                self.pybullet_client,
                                to_numpy(center),
                                to_numpy(radius),
                                orientation=to_numpy(obj_orientation),
                                color=color,
                            )
                        )
                    elif isinstance(single_primitive, MultiBoxField):
                        center, size = single_primitive.centers, single_primitive.sizes
                        if len(center) == 2:
                            center = torch.cat([center, torch.zeros(1, **self.tensor_args)])
                            size = torch.cat(
                                [size, default_height_2d * torch.ones(1, **self.tensor_args)]
                            )  # default height
                        center += obj_position
                        self.obstacles.append(
                            add_box(
                                self.pybullet_client,
                                to_numpy(center),
                                to_numpy(size / 2),
                                orientation=to_numpy(obj_orientation),
                                color=color,
                            )
                        )
                    else:
                        raise NotImplementedError(f"single_primitive={single_primitive} not implemented")

        # store obstacles
        self.pbompl_interface.set_obstacles(self.obstacles)

    def get_start_and_goal_states(
        self,
        q_pos_start=None,
        ee_pose_start=None,
        q_pos_goal=None,
        ee_pose_goal=None,
        n_joint_position_goal=1,
        sample_joint_position_goals_with_same_ee_pose=False,
        min_distance_q_pos_start_goal=0.0,
        debug=False,
    ):
        if q_pos_start is not None:
            assert self.pbompl_interface.is_state_valid(
                q_pos_start, check_bounds=True
            ), f"q_pos_start={q_pos_start} is NOT valid"
        if q_pos_goal is not None:
            assert self.pbompl_interface.is_state_valid(
                q_pos_goal, check_bounds=True
            ), f"q_pos_goal={q_pos_goal} is NOT valid"

        if q_pos_start is not None and q_pos_goal is not None:
            if np.linalg.norm(q_pos_start - q_pos_goal) < min_distance_q_pos_start_goal:
                print(f"q_pos_start={q_pos_start} and q_pos_goal={q_pos_goal} are too close")
                return [], []

        for i in tqdm(range(1000), disable=True):  # max tries
            if q_pos_start is not None:
                q_pos_start_tmp = copy(q_pos_start)
            else:
                q_pos_start_tmp = self.pbompl_interface.get_state_not_in_collision(
                    ee_pose_target=ee_pose_start, debug=debug
                )

            if q_pos_goal is not None:
                q_pos_goal_tmp = copy(q_pos_goal)
            else:
                q_pos_goal_tmp = self.pbompl_interface.get_state_not_in_collision(
                    ee_pose_target=ee_pose_goal, debug=debug
                )

            q_pos_goal_tmp_l = []
            # check if the distance between the start and goal states is greater than min_distance_q_pos_start_goal
            if np.linalg.norm(q_pos_start_tmp - q_pos_goal_tmp) < min_distance_q_pos_start_goal:
                print(f"{i}")
                continue

            if sample_joint_position_goals_with_same_ee_pose:
                # To generate several trajectories from the same joint position start state to
                # multiple joint position goals, resample a new goal state with the same end-effector pose
                ee_pose_goal = self.pbompl_interface.get_ee_pose(q_pos_goal_tmp, return_transformation=True)
                for _ in range(n_joint_position_goal):
                    try:
                        q_pos_tmp = self.pbompl_interface.get_state_not_in_collision(
                            ee_pose_target=ee_pose_goal, debug=debug
                        )
                        q_pos_goal_tmp_l.append(q_pos_tmp)
                    except:
                        pass
                assert len(q_pos_goal_tmp_l) > 0, f"len(joint_position_goal_l)={len(q_pos_goal_tmp_l)} == 0"
            else:
                q_pos_goal_tmp_l = [q_pos_goal_tmp] * n_joint_position_goal

            break

        # start state is the same for all goals
        q_pos_start_l = [q_pos_start_tmp] * len(q_pos_goal_tmp_l)

        return q_pos_start_l, q_pos_goal_tmp_l

    def run(
        self,
        num_trajectories,
        joint_position_start,
        joint_position_goal,
        planner_allowed_time=4.0,
        interpolate_num=250,
        simplify_path=True,
        fit_bspline=False,
        bspline_num_control_points=20,
        bspline_degree=5,
        bspline_zero_vel_at_start_and_goal=True,
        bspline_zero_acc_at_start_and_goal=True,
        max_tries=1000,
        duration_visualization=2.0,
        wait_time_after_visualization=4.0,
        debug=False,
    ):
        assert max_tries >= num_trajectories, (
            f"max_tries must be greater than the number of desired trajectories."
            f" max_tries={max_tries} < num_trajectories={num_trajectories}"
        )

        num_trajectories_generated = 0
        results_dict = {}

        # planning
        if debug:
            print(f"joint_position_start: {joint_position_start}")
            print(f"joint_position_goal: {joint_position_goal}")
        for i in range(max_tries):
            s_time = time.perf_counter()
            # set the internal robot state to the start state
            self.robot_pbompl.set_state(joint_position_start)
            results_dict_plan = self.pbompl_interface.plan(
                joint_position_goal,
                allowed_time=planner_allowed_time,
                interpolate_num=interpolate_num,
                simplify_path=simplify_path,
                fit_bspline=fit_bspline,
                bspline_num_control_points=bspline_num_control_points,
                bspline_degree=bspline_degree,
                bspline_zero_vel_at_start_and_goal=bspline_zero_vel_at_start_and_goal,
                bspline_zero_acc_at_start_and_goal=bspline_zero_acc_at_start_and_goal,
                debug=debug,
            )
            planning_time = time.perf_counter() - s_time
            if debug:
                print(f"planning time: {planning_time:.3f} s")

            if results_dict_plan["success"]:
                results_dict[num_trajectories_generated] = results_dict_plan

                num_trajectories_generated += 1
                if debug:
                    print(f"num_trajectories_generated: {num_trajectories_generated}/{num_trajectories}")

                if debug:
                    sol_path = results_dict_plan["sol_path"]
                    self.pbompl_interface.execute(sol_path, sleep_time=duration_visualization / len(sol_path))
                    time.sleep(wait_time_after_visualization)

            if num_trajectories_generated >= num_trajectories:
                break

        return results_dict

    def terminate(self):
        self.pybullet_client.disconnect()


def get_random_pose_from_region(pose_region):
    ee_pose_target = np.eye(4)
    # random position
    ee_pose_target[0, 3] = np.random.choice(
        [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["translation"]["x"]]
    )
    ee_pose_target[1, 3] = np.random.choice(
        [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["translation"]["y"]]
    )
    ee_pose_target[2, 3] = np.random.choice(
        [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["translation"]["z"]]
    )
    # random orientation
    rotation_base = Rotation.from_matrix(np.array(pose_region["rotation"]["base"]))
    rotation_random_around_base = Rotation.from_euler(
        "xyz",
        [
            np.random.choice(
                [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["rotation"]["x"]]
            ),
            np.random.choice(
                [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["rotation"]["y"]]
            ),
            np.random.choice(
                [np.random.rand(1).item() * (high - low) + low for low, high in pose_region["rotation"]["z"]]
            ),
        ],
        degrees=True,
    )
    rotation_target = rotation_base * rotation_random_around_base
    ee_pose_target[:3, :3] = rotation_target.as_matrix()
    return ee_pose_target


def get_random_ee_pose_from_cfg_file(env_id, robot_id, cfg_file_path):
    # create a random start and goal end-effector pose based on a config file describing the target pose limits
    with open(cfg_file_path, "r") as f:
        cfg_ee = yaml.load(f, Loader=yaml.Loader)

    assert (
        env_id == cfg_ee["env_id"] or env_id == cfg_ee["env_id"] + "ExtraObjects"
    ), f"env_id mismatch: {env_id} != {cfg_ee['env_id']}"
    assert robot_id == cfg_ee["robot_id"], f"robot_id mismatch: {robot_id} != {cfg_ee['robot_id']}"

    pose_region_ids = list(cfg_ee["pose_regions"].keys())

    ee_pose_start = None

    # randomly move between pose regions or go to one of the pose regions
    if cfg_ee["move_between_pose_regions"] and random.choice([True, False]):
        # move between two pose regions
        assert len(pose_region_ids) >= 2, f"len(pose_region_ids)={len(pose_region_ids)} < 2"
        pose_region_id_start, pose_region_id_goal = np.random.choice(pose_region_ids, size=2, replace=False)
        ee_pose_start = get_random_pose_from_region(cfg_ee["pose_regions"][pose_region_id_start])
        ee_pose_goal = get_random_pose_from_region(cfg_ee["pose_regions"][pose_region_id_goal])
    else:
        # move to one of the pose regions
        pose_region_id = np.random.choice(pose_region_ids)
        ee_pose_goal = get_random_pose_from_region(cfg_ee["pose_regions"][pose_region_id])

    return ee_pose_start, ee_pose_goal


def generate_trajectories_run(
    generate_data_ompl_worker,
    env_id,
    robot_id,
    planner,
    min_distance_robot_env,
    task_id,
    joint_position_start,
    joint_position_goal,
    planner_allowed_time,
    interpolate_num,
    simplify_path,
    fit_bspline,
    bspline_num_control_points,
    bspline_degree,
    bspline_zero_vel_at_start_and_goal,
    bspline_zero_acc_at_start_and_goal,
    tensor_args,
    num_trajectories=1,
    max_tries=1,
    pybullet_mode="DIRECT",
    debug=False,
):
    if generate_data_ompl_worker is None:
        # For multi-threading, create a new worker
        generate_data_ompl_worker = GenerateDataOMPL(
            env_id,
            robot_id,
            planner=planner,
            min_distance_robot_env=min_distance_robot_env,
            tensor_args=tensor_args,
            gripper=True,  # By default, to generate data, add the gripper to the robot
            pybullet_mode=pybullet_mode,
            debug=debug,
        )

    start_planning_time = time.time()  # start the timer only after the worker is created
    results_dict = generate_data_ompl_worker.run(
        num_trajectories=num_trajectories,  # each worker generates one trajectory
        max_tries=max_tries,
        joint_position_start=joint_position_start,
        joint_position_goal=joint_position_goal,
        planner_allowed_time=planner_allowed_time,
        interpolate_num=interpolate_num,
        fit_bspline=fit_bspline,
        simplify_path=simplify_path,
        bspline_num_control_points=bspline_num_control_points,
        bspline_degree=bspline_degree,
        bspline_zero_vel_at_start_and_goal=bspline_zero_vel_at_start_and_goal,
        bspline_zero_acc_at_start_and_goal=bspline_zero_acc_at_start_and_goal,
        debug=debug,
    )
    end_planning_time = time.time()

    # Add the task_id to all the results
    for k, v in results_dict.items():
        results_dict[k]["task_id"] = task_id

    # Update timing information
    results_dict.update(
        {
            "run_planning_time": end_planning_time - start_planning_time,
            "start_planning_time": start_planning_time,
            "end_planning_time": end_planning_time,
        }
    )

    return results_dict


@single_experiment_yaml
def experiment(
    ############################################################################
    # env_id: str = 'EnvDense2D',
    # env_id: str = 'EnvSimple2D',
    # env_id: str = 'EnvNarrowPassageDense2D',
    # robot_id: str = 'RobotPointMass2D',
    # env_id: str = 'EnvPlanar2Link',
    # robot_id: str = 'RobotPlanar2Link',
    # env_id: str = 'EnvPlanar4Link',
    # robot_id: str = 'RobotPlanar4Link',
    # env_id: str = 'EnvSpheres3D',
    # env_id: str = 'EnvSpheres3DExtraObjectsV00',
    # env_id: str = 'EnvTableShelf',
    # env_id: str = 'EnvPilars3D',
    # robot_id: str = 'RobotPanda',
    env_id: str = "EnvWarehouse",
    robot_id: str = "RobotPanda",
    ############################################################################
    start_task_id: int = 49400,
    num_tasks: int = 5,
    num_trajectories_per_task: int = 1,
    ############################################################################
    sample_joint_position_goals_with_same_ee_pose: bool = False,
    cfg_file: str = "None",
    # cfg_file: str = "EnvTableShelf-RobotPanda.yaml",
    # cfg_file: str = "EnvWarehouse-RobotPanda.yaml",
    # cfg_file: str = "EnvWarehouse-RobotPanda_v01.yaml",
    ############################################################################
    min_distance_robot_env: float = 0.00,
    min_distance_q_pos_start_goal: float = 0.0,  # minimum distance between start and goal joint positions
    # planner parameters
    # planner: str = "PRM",
    # planner: str = 'PRMstar',
    planner: str = "RRTConnect",
    # planner: str = 'RRTstar',
    # planner: str = 'BITstar',
    # planner: str = 'AITstar',
    planner_allowed_time: float = 10.0,
    # path simplification methods of OMPL
    simplify_path: bool = True,
    # bspline parameters
    fit_bspline: bool = False,
    bspline_num_control_points: int = 12,
    bspline_degree: int = 5,
    bspline_zero_vel_at_start_and_goal: bool = True,
    bspline_zero_acc_at_start_and_goal: bool = True,
    interpolate_num: int = 128,  # number of waypoints to interpolate the path
    #######################################
    n_parallel_jobs: int = 1,  # Set to 1 to debug with pybullet GUI
    # n_parallel_jobs: int = os.cpu_count(),
    debug: bool = True,
    #######################################
    # MANDATORY
    seed: int = int(time.time()),
    # seed: int = 49400,
    results_dir: str = f"data/env-robot",
    #######################################
    **kwargs,
):
    fix_random_seed(seed)

    print(f"\n\n-------------------- Generating data --------------------")
    print(f"Env:   {env_id}")
    print(f"Robot: {robot_id}")
    print(f"start_task_id:  {start_task_id}")
    print(f"num_tasks:  {num_tasks}")
    print(f"num_trajectories_per_task:  {num_trajectories_per_task}")
    print(f"\n\n--------------------------------------------------------")

    tensor_args = {"device": "cpu", "dtype": torch.float32}

    ####################################################################################################################
    # Create the tasks - start and goal joint positions or end-effector poses
    q_pos_start = None
    ee_pose_start = None
    q_pos_goal = None
    ee_pose_goal = None

    # pybullet can only run one GUI client
    pybullet_mode = "GUI" if debug and n_parallel_jobs == 1 else "DIRECT"
    generate_data_ompl_worker = GenerateDataOMPL(
        env_id,
        robot_id,
        planner=planner,
        min_distance_robot_env=min_distance_robot_env,
        tensor_args=tensor_args,
        gripper=True,  # By default, to generate data, add the gripper to the robot
        pybullet_mode=pybullet_mode,
        debug=debug,
    )

    print("\nGenerating tasks...")
    task_id_l = []
    joint_position_start_l = []
    joint_position_goal_l = []
    for i in tqdm(range(start_task_id, start_task_id + num_tasks)):

        if cfg_file != "None":
            # generate start and goal poses based on a config file
            cfg_file_path = os.path.join(DATA_GENERATION_CFGS_PATH, cfg_file)
            ee_pose_start, ee_pose_goal = get_random_ee_pose_from_cfg_file(env_id, robot_id, cfg_file_path)

        # print(f'joint_position_start: {joint_position_start}')
        # print(f'ee_pose_start: {ee_pose_start}')
        # print(f'joint_position_goal: {joint_position_goal}')
        # print(f'ee_pose_target: {ee_pose_goal}')

        # Sample start and goal states (joint positions or end-effector poses)
        try:
            # IK might fail to find a solution
            joint_position_start_l_tmp, joint_position_goal_l_tmp = generate_data_ompl_worker.get_start_and_goal_states(
                q_pos_start=q_pos_start,
                ee_pose_start=ee_pose_start,
                q_pos_goal=q_pos_goal,
                ee_pose_goal=ee_pose_goal,
                n_joint_position_goal=num_trajectories_per_task,
                sample_joint_position_goals_with_same_ee_pose=sample_joint_position_goals_with_same_ee_pose,
                min_distance_q_pos_start_goal=min_distance_q_pos_start_goal,
                debug=debug,
            )
        except Exception as e:
            print(e)
            continue

        task_id_l.extend([i] * len(joint_position_start_l_tmp))
        joint_position_start_l.extend(joint_position_start_l_tmp)
        joint_position_goal_l.extend(joint_position_goal_l_tmp)

    assert (
        len(task_id_l) == len(joint_position_start_l) == len(joint_position_goal_l)
    ), f"len(task_id_l)={len(task_id_l)} != len(joint_position_start_l)={len(joint_position_start_l)}"
    print(f"\n----------\nGenerated {len(task_id_l)}/{num_tasks} tasks successfully\n----------\n")

    ####################################################################################################################
    # Generate data
    # Generate data in parallel with joblib
    with TimerCUDA() as t_generate_data:
        results_dict_l = Parallel(n_jobs=n_parallel_jobs)(
            delayed(generate_trajectories_run)(
                generate_data_ompl_worker if n_parallel_jobs == 1 else None,
                env_id,
                robot_id,
                planner,
                min_distance_robot_env,
                task_id,
                joint_position_start,
                joint_position_goal,
                planner_allowed_time,
                interpolate_num,
                simplify_path,
                fit_bspline,
                bspline_num_control_points,
                bspline_degree,
                bspline_zero_vel_at_start_and_goal,
                bspline_zero_acc_at_start_and_goal,
                tensor_args,
                1,
                1,
                pybullet_mode,
                debug,
            )
            for task_id, joint_position_start, joint_position_goal in zip(
                task_id_l, joint_position_start_l, joint_position_goal_l
            )
        )

    generate_data_ompl_worker.terminate()

    ####################################################################################################################
    # Save timing information stats
    # grab timestamps from worker runs
    start_times = [r["start_planning_time"] for r in results_dict_l]
    end_times = [r["end_planning_time"] for r in results_dict_l]
    run_times = [r["run_planning_time"] for r in results_dict_l]

    # this is the window when actual planning computation happened
    planning_computation_time = max(end_times) - min(start_times)

    print("-" * 80)
    print(f"Total joblib wall time (incl. overhead): {t_generate_data.elapsed:.4f} sec")
    print(f"Actual parallel compute window: {planning_computation_time:.4f} sec")
    print(f"Ideal parallel time (max individual runtime): {max(run_times):.4f} sec")
    print(f"Total compute time (sum of all workers): {sum(run_times):.4f} sec")
    print(f"Parallel speedup: {sum(run_times) / planning_computation_time:.2f}x")

    with open(pathlib.Path(results_dir) / "timing_stats.pkl", "wb") as fp:
        pickle.dump(
            {
                "num_plans": len(task_id_l),
                "num_parallel_jobs": n_parallel_jobs,
                "joblib_wall_time": t_generate_data.elapsed,
                "planning_computation_time": planning_computation_time,
                "ideal_parallel_time": max(run_times),
                "total_compute_time": sum(run_times),
                "parallel_speedup": sum(run_times) / planning_computation_time,
            },
            fp,
        )

    ####################################################################################################################
    # Merge results for hdf5 format
    results_dict = {}
    num_trajectories_generated = 0
    for results_dict_run in results_dict_l:
        # drop if no trajectory was generated
        if len(results_dict_run) == 0:
            continue
        for k, v in results_dict_run.items():
            if k in ["run_planning_time", "start_planning_time", "end_planning_time"]:
                continue
            num_trajectories_generated += 1
            for kk, vv in v.items():
                if kk == "bspline_params" and vv is None:
                    continue
                elif kk == "bspline_params" and vv is not None:
                    tt, cc, k = vv
                    for x, y in zip(["bspline_params_tt", "bspline_params_cc", "bspline_params_k"], [tt, cc, k]):
                        if x in results_dict:
                            results_dict[x].append(y)
                        else:
                            results_dict[x] = [y]
                    continue

                if vv is None:
                    vv = False  # hdf5 does not support None

                if kk in results_dict:
                    results_dict[kk].append(vv)
                else:
                    results_dict[kk] = [vv]

    print("\n\n------------------------------------------------------------")
    num_trajectories_desired = num_tasks * num_trajectories_per_task
    print(
        f"Generated {num_trajectories_generated}/{num_trajectories_desired} trajectories"
        f" in {t_generate_data.elapsed:.3f} s"
    )

    # save results to disk
    hf = h5py.File(os.path.join(results_dir, "dataset.hdf5"), "w")
    for k, v in results_dict.items():
        hf.create_dataset(f"{k}", data=v, compression="gzip")
    # metadata
    hf.attrs["num_trajectories_desired"] = num_trajectories_desired
    hf.attrs["num_trajectories_generated"] = num_trajectories_generated
    hf.close()


if __name__ == "__main__":
    run_experiment(experiment)

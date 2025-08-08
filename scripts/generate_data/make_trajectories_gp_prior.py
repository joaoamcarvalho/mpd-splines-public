import os.path

import einops
import h5py
import numpy as np
import yaml
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm

from experiment_launcher.utils import fix_random_seed
from torch_robotics.environments import EnvEmpty2DExtraSquare
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


fix_random_seed(0)

ts = np.array([[0.0], [1.0]])

x_start_lims = np.array(
    [
        [-0.90, -0.85],
        [-0.025, 0.025],
    ]
)

x_goal_lims = np.array(
    [
        [0.85, 0.90],
        [-0.025, 0.025],
    ]
)

n_contexts = 1
n_trajectories_per_context = 100
n_interpolate = 256
length_scale_bound_lower = 1.1
ts_ = np.linspace(ts[0][0], ts[1][0], n_interpolate)[:, None]
trajectories_q = np.zeros((n_contexts, n_trajectories_per_context, n_interpolate, 2))
for i in tqdm(range(n_contexts)):
    # sample a start and goal configuration
    q_start = np.random.uniform(x_start_lims[:, 0], x_start_lims[:, 1])
    q_goal = np.random.uniform(x_goal_lims[:, 0], x_goal_lims[:, 1])
    xs = np.array([q_start, q_goal])

    # create a GP prior
    kernel = 1 * RBF(length_scale=length_scale_bound_lower, length_scale_bounds=(length_scale_bound_lower, 1e4))
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,
        n_restarts_optimizer=10,
    )
    gaussian_process.fit(ts, xs)

    # sample some trajectoroies
    xs_samples = gaussian_process.sample_y(ts_, n_samples=n_trajectories_per_context, random_state=None)
    xs_samples = np.moveaxis(xs_samples, -1, 0)
    trajectories_q[i] = xs_samples

trajectories_q_flat = einops.rearrange(trajectories_q, "... h d -> (...) h d")

# Save data to format compatible with the MPD library
args = dict(
    bspline_degree=5,
    bspline_num_control_points=16,
    bspline_zero_acc_at_start_and_goal=True,
    bspline_zero_vel_at_start_and_goal=True,
    cfg_file=None,
    debug=False,
    device="cpu",
    env_id="EnvEmpty2D",
    fit_bspline=False,
    interpolate_num=n_interpolate,
    min_distance_robot_env=0.0,
    n_parallel_jobs=1,
    num_trajectories_desired=n_contexts * n_trajectories_per_context,
    num_trajectories_generated=n_contexts * n_trajectories_per_context,
    num_trajectories_per_task=n_trajectories_per_context,
    planner="GPPrior",
    planner_allowed_time=10.0,
    robot_id="RobotPointMass2D",
    sample_joint_position_goals_with_same_ee_pose=False,
    selection="many" if n_trajectories_per_context > 1 else "one",
    simplify_path=True,
)

save_dir = os.path.join(f'./EnvEmpty2D-RobotPointMass2D-joint_joint-{args["selection"]}-GPPrior')
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "args.yaml"), "w") as fp:
    yaml.dump(args, fp)

# save results to disk
hf = h5py.File(os.path.join(save_dir, "dataset_merged.hdf5"), "w")
results_dict = dict(
    all_states_valid_after_bspline_fit=[False] * trajectories_q_flat.shape[0],
    sol_path=trajectories_q_flat,
    sol_path_after_bspline_fit=[False] * trajectories_q_flat.shape[0],
    success=[True] * trajectories_q_flat.shape[0],
    task_id=einops.repeat(np.arange(trajectories_q.shape[0]), "n -> (n b)", b=n_trajectories_per_context),
)
for k, v in results_dict.items():
    hf.create_dataset(f"{k}", data=v, compression="gzip")
# metadata
hf.attrs["num_trajectories_desired"] = args["num_trajectories_desired"]
hf.attrs["num_trajectories_generated"] = args["num_trajectories_generated"]
hf.close()

# plot trajectories
env = EnvEmpty2DExtraSquare(tensor_args=DEFAULT_TENSOR_ARGS)
fig, ax = create_fig_and_axes(env.dim)
env.render(ax)

for traj_q in trajectories_q_flat:
    ax.plot(traj_q[:, 0], traj_q[:, 1], "orange", alpha=0.5, zorder=-1)

plt.show()

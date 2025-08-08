import isaacgym

import os

import torch
from matplotlib import pyplot as plt

from experiment_launcher import single_experiment_yaml, run_experiment
from mpd import trainer
from mpd.models import UNET_DIM_MULTS, TemporalUnet
from mpd.models.diffusion_models.context_models import ContextModelQs, ContextModelEEPoseGoal, ContextModelCombined
from mpd.trainer.trainer import get_num_epochs
from mpd.utils.loaders import get_planning_task_and_dataset, get_model, get_loss, get_summary
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


os.environ["WANDB_API_KEY"] = "999"
WANDB_MODE = "disabled"
WANDB_ENTITY = "mpd-splines"
DEBUG = True


@single_experiment_yaml
def experiment(
    ########################################################################
    # Dataset
    dataset_subdir: str = "EnvSimple2D-RobotPointMass2D-joint_joint-one-RRTConnect",
    # dataset_subdir: str = 'EnvWarehouse-RobotPanda-config_file_v01-joint_joint-one-RRTConnect',
    dataset_file_merged: str = "dataset_merged_doubled.hdf5",
    reload_data: bool = False,
    preload_data_to_device: bool = False,
    n_task_samples: int = -1,  # -1 for all
    ########################################################################
    # Parametric trajectory
    parametric_trajectory_class: str = "ParametricTrajectoryBspline",
    # parametric_trajectory_class: str = 'ParametricTrajectoryWaypoints',
    bspline_degree: int = 5,
    bspline_num_control_points_desired: int = 22,  # adjusted such that trainable control points are a multiple of 8
    num_T_pts: int = 128,  # number of time steps for trajectory interpolation
    ########################################################################
    # Context model
    # condition on joint start and goal
    context_qs: bool = True,
    context_qs_n_layers: int = 2,
    context_q_out_dim: int = 128,
    context_qs_act: str = "relu",
    # End-effector pose conditioned model
    context_ee_goal_pose: bool = False,
    context_ee_goal_pose_n_layers: int = 2,
    context_ee_goal_pose_out_dim: int = 128,
    context_ee_goal_pose_act: str = "relu",
    # Combined context model
    context_combined_out_dim: int = 128,
    ########################################################################
    # Generative prior model
    generative_model_class: str = "GaussianDiffusionModel",  # 'GaussianDiffusionModel', 'CVAEModel'
    # Diffusion Model
    variance_schedule: str = "cosine",
    n_diffusion_steps: int = 100,
    predict_epsilon: bool = True,
    conditioning_type: str = "default",  # 'default', 'concatenate', 'attention'
    # Unet
    unet_input_dim: int = 32,
    unet_dim_mults_option: int = 1,
    # CVAE
    cvae_latent_dim: int = 32,
    loss_cvae_kl_weight: float = 1e-1,
    ########################################################################
    # Training parameters
    batch_size: int = 128,
    lr: float = 3e-4,
    clip_grad: bool = False,
    num_train_steps: int = 1_000_000,
    use_ema: bool = True,
    use_amp: bool = False,
    # Summary parameters
    steps_til_summary: int = 5000 if DEBUG else 20000,
    summary_class: str = "SummaryTrajectoryGeneration",
    steps_til_ckpt: int = 5000 if DEBUG else 20000,
    ########################################################################
    device: str = "cuda:0",
    debug: bool = DEBUG,
    ########################################################################
    # MANDATORY
    # seed: int = int(time.time()),
    seed: int = 1726484688,
    results_dir: str = "logs",
    ########################################################################
    # WandB
    wandb_mode: str = "disabled" if DEBUG else WANDB_MODE,  # "online", "offline" or "disabled"
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = "test_train_bspline_diffusion",
    **kwargs,
):
    print()
    print("-" * 100)
    print(f"{dataset_subdir} -- {parametric_trajectory_class}")
    print("-" * 100)
    print()

    # Set random seed for reproducibility
    fix_random_seed(seed)

    device = get_torch_device(device=device)
    tensor_args = {"device": device, "dtype": torch.float32}

    ########################################################################
    # Planning task and dataset
    planning_task, train_subset, train_dataloader, val_subset, val_dataloader = get_planning_task_and_dataset(
        parametric_trajectory_class=parametric_trajectory_class,
        dataset_subdir=dataset_subdir,
        dataset_file_merged=dataset_file_merged,
        reload_data=reload_data,
        preload_data_to_device=preload_data_to_device,
        n_task_samples=n_task_samples,
        bspline_degree=bspline_degree,
        bspline_num_control_points_desired=bspline_num_control_points_desired,
        num_T_pts=num_T_pts,
        context_qs=context_qs,
        context_ee_goal_pose=context_ee_goal_pose,
        batch_size=batch_size,
        results_dir=results_dir,
        save_indices=True,
        tensor_args=tensor_args,
    )

    full_dataset = train_subset.dataset

    if debug:
        full_dataset.render(
            task_id=0,
            render_joint_trajectories=True,
            render_robot_trajectories=True if full_dataset.planning_task.env.dim == 2 else False,
            render_n_robot_trajectories=50,
        )
        plt.show()

    ########################################################################
    # Model
    context_model_qs = None
    if context_qs:
        context_model_qs = ContextModelQs(
            in_dim=full_dataset.context_q_dim,
            out_dim=context_q_out_dim,
            n_layers=context_qs_n_layers,
            act=context_qs_act,
        )

    context_model_ee_pose_goal = None
    if context_ee_goal_pose:
        context_model_ee_pose_goal = ContextModelEEPoseGoal(
            out_dim=context_ee_goal_pose_out_dim,
            n_layers=context_ee_goal_pose_n_layers,
            act=context_ee_goal_pose_act,
        )

    context_model = None
    if not (context_model_qs is None and context_model_ee_pose_goal is None):
        context_model = ContextModelCombined(
            context_model_qs=context_model_qs,
            context_model_ee_pose_goal=context_model_ee_pose_goal,
            out_dim=context_combined_out_dim,
        )

    diffusion_configs = dict(
        variance_schedule=variance_schedule,
        n_diffusion_steps=n_diffusion_steps,
        predict_epsilon=predict_epsilon,
    )

    cvae_configs = dict(
        cvae_latent_dim=cvae_latent_dim,
    )

    unet_configs = dict(
        state_dim=full_dataset.state_dim,
        n_support_points=full_dataset.n_learnable_control_points,
        unet_input_dim=unet_input_dim,
        dim_mults=UNET_DIM_MULTS[unet_dim_mults_option],
        conditioning_type=conditioning_type if context_model is not None else "None",
        conditioning_embed_dim=context_model.out_dim if context_model is not None else None,
    )

    model = get_model(
        model_class=generative_model_class,
        denoise_fn=TemporalUnet(**unet_configs),
        context_model=context_model,
        tensor_args=tensor_args,
        **cvae_configs,
        **diffusion_configs,
        **unet_configs,
    )

    ########################################################################
    # Loss
    if generative_model_class == "GaussianDiffusionModel":
        loss_class = "GaussianDiffusionLoss"
    elif generative_model_class == "CVAEModel":
        loss_class = "CVAELoss"
    else:
        raise ValueError(f"Unknown generative_model_class: {generative_model_class}")

    loss_fn = val_loss_fn = get_loss(loss_class=loss_class, loss_cvae_kl_weight=loss_cvae_kl_weight)

    ########################################################################
    # Summary
    summary_fn = get_summary(
        summary_class=summary_class,
        debug=debug,
    )

    ########################################################################
    # Train
    trainer.train(
        model=model,
        train_dataloader=train_dataloader,
        train_subset=train_subset,
        val_dataloader=val_dataloader,
        val_subset=val_subset,
        planning_task=planning_task,
        num_train_steps=num_train_steps,
        epochs=get_num_epochs(num_train_steps, batch_size, len(train_subset)),
        model_dir=results_dir,
        summary_fn=summary_fn,
        lr=lr,
        lr_scheduler=False,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        steps_til_summary=steps_til_summary,
        steps_til_checkpoint=steps_til_ckpt,
        clip_grad=clip_grad,
        early_stopper_patience=-1,
        use_ema=use_ema,
        use_amp=use_amp,
        debug=debug,
        tensor_args=tensor_args,
    )


if __name__ == "__main__":
    # Leave unchanged
    run_experiment(experiment)

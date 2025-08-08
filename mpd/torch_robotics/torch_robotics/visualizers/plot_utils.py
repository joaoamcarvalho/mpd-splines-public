import os
from copy import copy

import numpy as np
import torch
from matplotlib import pyplot as plt, collections as mcoll
from matplotlib.animation import FuncAnimation

from torch_robotics.torch_utils.torch_utils import to_numpy, DEFAULT_TENSOR_ARGS


def plot_coordinate_frame(
    ax, frame, arrow_length=0.1, arrow_alpha=1.0, arrow_linewidth=1.0, tensor_args=DEFAULT_TENSOR_ARGS
):
    target_pos_np = to_numpy(frame.translation).squeeze()
    frame_target_no_trans = copy(frame)
    frame_target_no_trans.set_translation(torch.zeros_like(frame.translation))

    x_basis = torch.tensor([1, 0, 0], **tensor_args)
    y_basis = torch.tensor([0, 1, 0], **tensor_args)
    z_basis = torch.tensor([0, 0, 1], **tensor_args)

    x_axis_target = to_numpy(frame_target_no_trans.transform_point(x_basis).squeeze())
    y_axis_target = to_numpy(frame_target_no_trans.transform_point(y_basis).squeeze())
    z_axis_target = to_numpy(frame_target_no_trans.transform_point(z_basis).squeeze())

    # x-axis
    ax.quiver(
        target_pos_np[0],
        target_pos_np[1],
        target_pos_np[2],
        x_axis_target[0],
        x_axis_target[1],
        x_axis_target[2],
        length=arrow_length,
        normalize=True,
        color="red",
        alpha=arrow_alpha,
        linewidth=arrow_linewidth,
    )
    # y-axis
    ax.quiver(
        target_pos_np[0],
        target_pos_np[1],
        target_pos_np[2],
        y_axis_target[0],
        y_axis_target[1],
        y_axis_target[2],
        length=arrow_length,
        normalize=True,
        color="green",
        alpha=arrow_alpha,
        linewidth=arrow_linewidth,
    )
    # z-axis
    ax.quiver(
        target_pos_np[0],
        target_pos_np[1],
        target_pos_np[2],
        z_axis_target[0],
        z_axis_target[1],
        z_axis_target[2],
        length=arrow_length,
        normalize=True,
        color="blue",
        alpha=arrow_alpha,
        linewidth=arrow_linewidth,
    )


def create_fig_and_axes(dim=2, figsize=(8, 6)):
    if dim == 3:
        fig = plt.figure(layout="tight")
        ax = fig.add_subplot(projection="3d")
    else:
        fig, ax = plt.subplots(figsize=figsize)

    return fig, ax


def create_animation_video(
    fig, animate_fn, anim_time=5, n_frames=100, video_filepath="video.mp4", dpi=90, fargs=(), **kwargs
):
    str_start = "Creating animation"
    print(f"{str_start}...")
    ani = FuncAnimation(
        fig, animate_fn, frames=n_frames, fargs=fargs, interval=anim_time * 1000 / n_frames, repeat=False
    )
    print(f"...finished {str_start}")

    str_start = "Saving video"
    print(f"{str_start}...")
    ani.save(os.path.join(video_filepath), fps=max(1, int(n_frames / anim_time)), dpi=dpi)
    print(f"...finished {str_start}")


def plot_multiline(ax, X, Y, color="blue", linestyle="solid", plot_scatter_points=False, **kwargs):
    segments = np.stack((X, Y), axis=-1)
    line_segments = mcoll.LineCollection(segments, colors=[color] * len(segments), linestyle=linestyle)
    ax.add_collection(line_segments)

    # Manually adjust the axis limits to fit the data
    all_points = np.array(segments).reshape(-1, 2)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    # Set limits with some padding
    padding = 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    if plot_scatter_points:
        points = np.reshape(segments, (-1, 2))
        ax.scatter(points[:, 0], points[:, 1], color=color, s=2**2)

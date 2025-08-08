import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
import torch_robotics.robots as tr_robots
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvEmpty2D(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, precompute_sdf_obj_fixed=True, sdf_cell_size=0.005, **kwargs):

        obj_list = [
            MultiSphereField(
                np.array(
                    [
                        [0, 0],
                    ]
                ),
                np.array(
                    [
                        0.0001,
                    ]
                ),  # using a tiny radius to create a dummy object
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, "empty2d")],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs,
        )


class EnvEmpty2DExtraSphere(EnvEmpty2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiSphereField(
                np.array(
                    [
                        [0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        0.3,
                    ]
                ),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "empty2d-extra-sphere")], tensor_args=tensor_args, **kwargs
        )


class EnvEmpty2DExtraNonConvex(EnvEmpty2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.0, -0.2],
                        [-0.2, 0.0],
                        [0.2, 0.0],
                    ]
                ),
                np.array(
                    [
                        [0.5, 0.1],
                        [0.1, 0.4],
                        [0.1, 0.4],
                    ]
                ),
                tensor_args=tensor_args,
            )
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "empty2d-extra-nonconvex")], tensor_args=tensor_args, **kwargs
        )


class EnvEmpty2DExtraSquare(EnvEmpty2D):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        obj_extra_list = [
            MultiBoxField(
                np.array(
                    [
                        [0.0, 0.0],
                    ]
                ),
                np.array(
                    [
                        [0.6, 0.6],
                    ]
                ),
                tensor_args=tensor_args,
            ),
        ]

        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "empty2d-extra-square")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvEmpty2D(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    ##############################
    env = EnvEmpty2DExtraSphere(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    ##############################
    env = EnvEmpty2DExtraNonConvex(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

    ##############################
    env = EnvEmpty2DExtraSquare(tensor_args=DEFAULT_TENSOR_ARGS)
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

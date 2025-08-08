import numpy as np
import torch
from matplotlib import pyplot as plt

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import MultiSphereField, ObjectField
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.plot_utils import create_fig_and_axes


class EnvPlanar4Link(EnvBase):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        # x, y, radius
        circles = np.array(
            [
                (-0.5, 0.5, 0.15),
                (0.0, 0.65, 0.2),
                (0.6, 0.3, 0.15),
                (0.5, -0.1, 0.15),
                (0.0, -0.5, 0.15),
                (-0.5, -0.1, 0.2),
            ]
        )
        spheres = MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)
        obj_field = ObjectField([spheres], "planar4link-spheres")
        obj_list = [obj_field]

        super().__init__(
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=obj_list,
            tensor_args=tensor_args,
            **kwargs,
        )


class EnvPlanar4LinkExtraObjectsV00(EnvPlanar4Link):

    def __init__(self, tensor_args=DEFAULT_TENSOR_ARGS, **kwargs):
        # x, y, radius
        circles = np.array(
            [
                (0.375, 0.5, 0.1),
                (0.3, -0.35, 0.1),
                (-0.3, -0.35, 0.1),
                (-0.6, 0.25, 0.1),
            ]
        )
        obj_extra_list = [MultiSphereField(circles[:, :2], circles[:, 2], tensor_args=tensor_args)]
        super().__init__(
            obj_extra_list=[ObjectField(obj_extra_list, "planar4link-extraobjects")], tensor_args=tensor_args, **kwargs
        )


if __name__ == "__main__":
    env = EnvPlanar4Link(precompute_sdf_obj_fixed=True, tensor_args=DEFAULT_TENSOR_ARGS)
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
    env = EnvPlanar4LinkExtraObjectsV00(
        precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()

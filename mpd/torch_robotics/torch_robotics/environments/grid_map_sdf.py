from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd.functional import jacobian
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class GridMapSDF:
    """
    Generates an SDF grid.
    """

    def __init__(self, limits, cell_size, obj_list, batch_size=64, tensor_args=DEFAULT_TENSOR_ARGS):

        self.limits = limits
        self.dim = limits.shape[-1]

        self.tensor_args = tensor_args

        # objects
        self.obj_list = obj_list

        # SDF grids
        map_dim = torch.abs(limits[1] - limits[0])
        self.map_dim = map_dim
        self.cell_size = cell_size
        self.cmap_dim = torch.ceil(map_dim / cell_size).type(torch.LongTensor).to(self.tensor_args["device"])

        self.points_for_sdf = None
        self.sdf_tensor = None
        self.grad_sdf_tensor = None

        self.batch_size = batch_size
        self.precompute_sdf()

    def precompute_sdf(self):
        # computes the signed distance field and its gradient
        # create voxel grid of points
        basis_ranges = [
            torch.linspace(self.limits[0][0], self.limits[1][0], self.cmap_dim[0], **self.tensor_args),
            torch.linspace(self.limits[0][1], self.limits[1][1], self.cmap_dim[1], **self.tensor_args),
        ]
        if self.dim == 3:
            basis_ranges.append(
                torch.linspace(self.limits[0][2], self.limits[1][2], self.cmap_dim[2], **self.tensor_args)
            )
        points_for_sdf_meshgrid = torch.meshgrid(*basis_ranges, indexing="ij")
        self.points_for_sdf = torch.stack(points_for_sdf_meshgrid, dim=-1)

        # compute sdf
        # if there are memory errors, use chunk_size=self.batch_size
        compute_sdf_raw_vmap = torch.vmap(self.compute_signed_distance_raw)
        self.sdf_tensor = compute_sdf_raw_vmap(self.points_for_sdf)

        # compute sdf gradient
        f_grad_sdf = lambda x: self.compute_signed_distance_raw(x).sum()
        self.grad_sdf_tensor = torch.func.jacrev(f_grad_sdf)(self.points_for_sdf)

    def compute_signed_distance_raw(self, x):
        sdf = None
        for obj in self.obj_list:
            sdf_obj = obj.compute_signed_distance(x)
            if sdf is None:
                sdf = sdf_obj
            else:
                sdf = torch.minimum(sdf, sdf_obj)
        return sdf

    def project_x_to_grid_points(self, X, **kwargs):
        # Project X to grid points
        X_in_map = ((X - self.limits[0]) / self.map_dim * self.cmap_dim).round().type(torch.LongTensor)

        # Project out-of-bounds locations to axis
        max_idx = torch.tensor(self.points_for_sdf.shape[:-1]) - 1
        X_in_map = X_in_map.clamp(torch.zeros_like(max_idx), max_idx)
        return X_in_map

    def __call__(self, X, **kwargs):
        X_in_map = self.project_x_to_grid_points(X, **kwargs)
        return self.get_sdf(X, X_in_map, **kwargs)

    def compute_cost(self, X, **kwargs):
        X_in_map = self.project_x_to_grid_points(X, **kwargs)
        return self.get_sdf(X, X_in_map, **kwargs)

    def compute_signed_distance(self, X, get_gradient=False, **kwargs):
        X_in_map = self.project_x_to_grid_points(X, **kwargs)

        if get_gradient:
            return self.get_sdf_and_gradient(X_in_map, **kwargs)
        else:
            return self.get_sdf(X, X_in_map, **kwargs)

    def get_sdf(self, X, X_in_map, **kwargs):
        """
        Checks for collision in a batch of trajectories using the generated occupancy grid (i.e. obstacle map), and
        returns sum of collision costs for the entire batch.

        :param weight: weight on obstacle cost, float tensor.
        :param X: Tensor of trajectories, of shape (batch_size, horizon, task_spaces, position_dim)
        :return: collision cost on the trajectories
        """
        # SDFs and gradients
        # To compute the gradients, because we already have computed the gradients, we use a surrogate sdf function
        # surrogate_sdf(x) = sdf(x_detachted) + x @ grad_sdf(x_detachted) - x_detached @ grad_sdf(x_detachted)
        # This surrogate has the property
        # surrogate_sdf(x) = sdf(x_detachted)
        # grad_x_surrogate_sdf(x) = grad_sdf(x_detachted)
        try:
            X_in_map_detached = X_in_map.detach()
            X_query = X_in_map_detached[..., 0], X_in_map_detached[..., 1]
            if self.dim == 3:
                X_query = X_in_map_detached[..., 0], X_in_map_detached[..., 1], X_in_map_detached[..., 2]

            sdf_vals = self.sdf_tensor[X_query]
            grad_sdf = self.grad_sdf_tensor[X_query]

            X_detached = X.detach()
            sdf_vals += (X * grad_sdf).sum(-1) - (X_detached * grad_sdf).sum(-1)

        except Exception as e:
            print(e)
            print(X_in_map)

        return sdf_vals

    def get_sdf_and_gradient(self, X_in_map, **kwargs):
        if self.dim == 2:
            X_query = X_in_map[..., 0], X_in_map[..., 1]
        elif self.dim == 3:
            X_query = X_in_map[..., 0], X_in_map[..., 1], X_in_map[..., 2]
        else:
            raise ValueError("Invalid dimension")

        sdf_vals = self.sdf_tensor[X_query]
        grad_sdf = self.grad_sdf_tensor[X_query]

        return sdf_vals, grad_sdf

    def zero_grad(self):
        pass

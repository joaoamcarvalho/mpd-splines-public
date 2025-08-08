import torch

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


class UnaryFactor:

    def __init__(
        self,
        dim,
        sigma,
        mean=None,
        tensor_args=DEFAULT_TENSOR_ARGS,
    ):
        self.sigma = sigma
        if mean is None:
            self.mean = torch.zeros(dim, **tensor_args)
        else:
            self.mean = mean
        self.tensor_args = tensor_args
        self.K = torch.eye(dim, **tensor_args) / sigma**2  # weight matrix
        self.dim = dim

    def get_error(self, x, calc_jacobian=True):
        error = self.mean - x

        if calc_jacobian:
            H = torch.eye(self.dim, **self.tensor_args).unsqueeze(0).repeat(x.shape[0], 1, 1)
            return error.view(x.shape[0], self.dim, 1), H
        else:
            return error

    def set_mean(self, x):
        self.mean = x.clone().detach()

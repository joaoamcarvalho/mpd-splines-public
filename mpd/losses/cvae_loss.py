import torch
import wandb
from torch import nn


class CVAELoss:

    def __init__(self, loss_cvae_kl_weight=1e-2, **kwargs):
        self.loss_cvae_kl_weight = loss_cvae_kl_weight

    def loss_fn(self, cvae_model, input_dict, dataset, step=None, **kwargs):
        """
        Loss function for training CVAE generative models.
        """
        control_points_normalized = input_dict[f"{dataset.field_key_control_points}_normalized"]

        loss, info = cvae_model.loss(
            control_points_normalized, input_dict, loss_cvae_kl_weight=self.loss_cvae_kl_weight, **kwargs
        )

        loss_dict = {"cvae_loss": loss}

        return loss_dict, info

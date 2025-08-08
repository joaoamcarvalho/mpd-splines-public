import torch


class GaussianDiffusionLoss:

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def loss_fn(diffusion_model, input_dict, dataset, step=None):
        """
        Loss function for training diffusion-based generative models.
        """
        control_points_normalized = input_dict[f"{dataset.field_key_control_points}_normalized"]

        hard_conds = input_dict.get("hard_conds", {})
        loss, _ = diffusion_model.loss(control_points_normalized, input_dict, hard_conds)

        loss_dict = {"diffusion_loss": loss}
        info = {"diffusion_loss": loss}

        return loss_dict, info

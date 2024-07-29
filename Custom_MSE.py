""" By Lennard Rose 5112737"""

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class MSE_with_penalty(_Loss):
    """
    custom loss function to penalize misclassifying the minority class (1).
    Applies a penalty in form of multiplying the loss when misclassifying a 1 by a factor.
    """
    def __init__(self, penalty=10):
        super().__init__()
        # multiply target by factor so the mean square error for the 1s is factor times bigger, the mse for 0 stays the same
        self.penalty = penalty


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        sq_diff = 0
        if len(input.shape) >= 2:
            for target_batch, input_batch in zip(target, input):
                for y, y_hat in zip(target_batch, input_batch):
                    factor = self.penalty if y == 1 else 1
                    sq_diff += (y - y_hat) ** 2 * factor
        else: # one dimensional batches -> just one output not n_to_n
            for y, y_hat in zip(target, input):
                factor = self.penalty if y == 1 else 1
                sq_diff += (y - y_hat) ** 2 * factor
        return torch.div(sq_diff, torch.numel(input))

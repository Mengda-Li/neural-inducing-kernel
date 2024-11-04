# loss/ssim_loss.py

import torch
import kornia
from .abstract_loss import Loss

class SSIMLoss(Loss):
    def __init__(self, window_size: int = 11, max_val: float = 255.0):
        """
        Initialize SSIMLoss.

        Args:
            window_size (int): Size of the Gaussian window for SSIM. Default is 11.
            max_val (float): Maximum pixel value in the image. Default is 1.0 for normalized images.
        """
        self.window_size = window_size
        self.max_val = max_val

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSIM loss as 1 - SSIM between prediction and target.

        Args:
            prediction (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The ground truth target tensor.

        Returns:
            torch.Tensor: Computed SSIM loss.
        """
        return kornia.losses.ssim_loss(prediction, target, window_size=self.window_size, max_val=self.max_val)

# metrics/ssim_metric.py

import torch
import kornia
from .metric import Metric

class SSIMMetric(Metric):
    def __init__(self, window_size: int = 11, max_val: float = 255.0):
        """
        Initialize SSIM metric.

        Args:
            window_size (int): Size of the Gaussian window for SSIM. Default is 11.
            max_val (float): Maximum pixel value in the image. Default is 255.0 for RGB image; (1.0 for normalized images).
        """
        self.window_size = window_size
        self.max_val = max_val

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the SSIM score as the average of SSIM tensor between prediction and target.

        Args:
            prediction (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The ground truth target tensor.

        Returns:
            torch.Tensor: Computed SSIM score.
        """
        ssim_score = kornia.metrics.ssim(prediction, target, window_size=self.window_size, max_val=self.max_val)
        return ssim_score.mean()  
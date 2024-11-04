# metrics/metric.py

from abc import ABC, abstractmethod
import torch

class Metric(ABC):
    @abstractmethod
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the score between the prediction and target.

        Args:
            prediction (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The ground truth target tensor.

        Returns:
            torch.Tensor: Computed score value.
        """

        pass
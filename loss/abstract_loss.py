# loss/loss.py

from abc import ABC, abstractmethod
import torch

class Loss(ABC):
    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between the prediction and target.

        Args:
            prediction (torch.Tensor): The predicted output tensor.
            target (torch.Tensor): The ground truth target tensor.

        Returns:
            torch.Tensor: Computed loss value.
        """
        pass

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(prediction, target)

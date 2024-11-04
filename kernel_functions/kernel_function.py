import torch
from abc import ABC, abstractmethod

class KernelFunction(ABC):

    @abstractmethod
    def compute(self, x, y):
        """
        Abstract method to compute the kernel function value between x and y.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 3, num_induce_pt).
            y (torch.Tensor): Tensor of shape (batch_size, 3, H, W).
            
        Returns:
            torch.Tensor: Kernel values with shape (batch_size, num_induce_pt, H, W).
        """
        pass
    
    def __call__(self, x_batch, alpha_batch):
        """
        Returns a function f(y) that computes the weighted sum of kernel values.

        Args:
            x_batch (torch.Tensor): Batch of inducing points, shape (batch_size, 3, num_induce_pt).
            alpha_batch (torch.Tensor): Corresponding kernel weights, shape (batch_size, 3, num_induce_pt).

        Returns:
            function: A function f(y) that computes f(y) = sum_i alpha_i * K(x_i, y) for each channel. So the final output is with shape (batch_size, 3, H, W)
        """
        def f(y):
            # Compute all kernel values in parallel between each x_i in x_batch and y
            kernel_values = self.compute(x_batch, y)  # Shape: (batch_size, num_induce_pt, H, W)
            
            # Weighted sum over the inducing points (num_induce_pt) using einsum
            # The einsum string "bci,bihw->bchw" allows summing across num_induce_pt for each channel
            output = torch.einsum('bci,bihw->bchw', alpha_batch, kernel_values)
            return output
        
        return f

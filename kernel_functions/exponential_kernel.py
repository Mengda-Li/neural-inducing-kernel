import torch
from .kernel_function import KernelFunction

class ExponentialKernel(KernelFunction):
    def __init__(self, sigma=1.0):
        """
        Exponential kernel function.
        
        Args:
            sigma (float): Standard deviation parameter, controls the width of the kernel.
        """
        self.sigma = sigma
        self.gamma = 1 / (2 * sigma ** 2)
    
    def compute(self, x, y):
        """
        Computes the exponential kernel function values between each x_i in x_batch and y.

        Args:
            x (torch.Tensor): Tensor of shape (batch_size, 3, num_induce_pt).
            y (torch.Tensor): Tensor of shape (batch_size, 3, H, W).
        
        Returns:
            torch.Tensor: Kernel values for each inducing point in x_batch, shape (batch_size, num_induce_pt, H, W).
        """
        # Reshape x for broadcasting: (batch_size, 3, num_induce_pt, 1, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 3, num_induce_pt, 1, 1)
        
        # Broadcast y for computation with each inducing point in x
        y = y.unsqueeze(2)  # Shape: (batch_size, 3, 1, H, W)
        
        # Compute the squared Euclidean distance without taking square root
        # Resulting shape after summing over the RGB channels: (batch_size, num_induce_pt, H, W)
        squared_distance = torch.sum(torch.square(x - y), dim=1)  # Shape: (batch_size, num_induce_pt, H, W)
        
        # Apply the exponential kernel function
        return torch.exp(-squared_distance * self.gamma)

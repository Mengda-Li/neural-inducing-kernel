# kernel_functions/_debug.py

from kernel_functions import ExponentialKernel
import torch

if __name__ == "__main__":
    # Parameters
    batch_size = 4
    num_induce_pt = 5
    H, W = 224, 224
    sigma = 1.0
    
    # Instantiate the exponential kernel
    exp_kernel = ExponentialKernel(sigma=sigma)
    
    # Batch of inducing points and weights
    x_batch = torch.rand(batch_size, 3, num_induce_pt)  # Shape: (batch_size, 3, num_induce_pt)
    alpha_batch = torch.rand(batch_size, 3, num_induce_pt)  # Shape: (batch_size, 3, num_induce_pt)
    
    # Create the function f
    f = exp_kernel(x_batch, alpha_batch)
    
    # Compute f(y) for an input image batch
    y = torch.rand(batch_size, 3, H, W)  # Shape: (batch_size, 3, H, W)
    result = f(y)  # Expected output shape: (batch_size, 3, H, W)
    print("Output shape:", result.shape)  # Should print (batch_size, 3, H, W)

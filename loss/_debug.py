import torch
from loss import SSIMLoss, DeltaE_CIEDE2000_Loss

# Example tensors (batch_size, channels, height, width)
prediction = torch.rand(4, 3, 224, 224)  # Random predicted image
target = torch.rand(4, 3, 224, 224)  # Random target image

# Initialize losses
ssim_loss = SSIMLoss(window_size=11, max_val=1.0)
delta_e_loss = DeltaE_CIEDE2000_Loss()

# Calculate losses
ssim_loss_value = ssim_loss(prediction, target)
delta_e_loss_value = delta_e_loss(prediction, target)

print("SSIM Loss:", ssim_loss_value.item())
print("Delta E CIEDE2000 Loss:", delta_e_loss_value.item())

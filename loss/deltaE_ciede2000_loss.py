# loss/deltaE_ciede2000_loss.py

import torch
from .abstract_loss import Loss
import math
import kornia

def deltaE_ciede2000(lab1: torch.Tensor, lab2: torch.Tensor, kL=1, kC=1, kH=1) -> torch.Tensor:
    """
    Calculate CIEDE2000 color difference between two LAB images.

    Args:
        lab1 (torch.Tensor): Reference LAB color tensor, shape (..., 3).
        lab2 (torch.Tensor): Comparison LAB color tensor, shape (..., 3).
        kL (float): Lightness scale factor, usually 1.
        kC (float): Chroma scale factor, usually 1.
        kH (float): Hue scale factor, usually 1.

    Returns:
        torch.Tensor: CIEDE2000 color difference.
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    
    C_avg = (C1 + C2) / 2
    G = 0.5 * (1 - torch.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = torch.sqrt(a1_prime**2 + b1**2)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = torch.atan2(b1, a1_prime) % (2 * math.pi)
    h2_prime = torch.atan2(b2, a2_prime) % (2 * math.pi)
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = torch.where(delta_h_prime > math.pi, delta_h_prime - 2 * math.pi, delta_h_prime)
    delta_h_prime = torch.where(delta_h_prime < -math.pi, delta_h_prime + 2 * math.pi, delta_h_prime)
    delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime / 2)
    
    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    
    h_avg_prime = (h1_prime + h2_prime) / 2
    h_avg_prime = torch.where(torch.abs(h1_prime - h2_prime) > math.pi, h_avg_prime - math.pi, h_avg_prime)
    h_avg_prime = h_avg_prime % (2 * math.pi)
    
    T = (1 
         - 0.17 * torch.cos(h_avg_prime - math.radians(30))
         + 0.24 * torch.cos(2 * h_avg_prime)
         + 0.32 * torch.cos(3 * h_avg_prime + math.radians(6))
         - 0.20 * torch.cos(4 * h_avg_prime - math.radians(63)))
    
    SL = 1 + 0.015 * ((L_avg_prime - 50) ** 2) / torch.sqrt(20 + (L_avg_prime - 50) ** 2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T
    
    delta_theta = math.radians(30) * torch.exp(-((h_avg_prime - math.radians(275)) / math.radians(25)) ** 2)
    RC = 2 * torch.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
    
    RT = -torch.sin(2 * delta_theta) * RC
    
    delta_E = torch.sqrt(
        (delta_L_prime / (kL * SL)) ** 2
        + (delta_C_prime / (kC * SC)) ** 2
        + (delta_H_prime / (kH * SH)) ** 2
        + RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    
    return delta_E

class DeltaE_CIEDE2000_Loss(Loss):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Delta E CIEDE2000 color difference loss.

        Args:
            prediction (torch.Tensor): The predicted RGB tensor, normalized to [0, 1].
            target (torch.Tensor): The target RGB tensor, normalized to [0, 1].

        Returns:
            torch.Tensor: Computed Delta E CIEDE2000 loss.
        """
        # Convert RGB to LAB
        prediction_lab = kornia.color.rgb_to_lab(prediction)
        target_lab = kornia.color.rgb_to_lab(target)

        # Compute color difference in CIEDE2000 space
        delta_e = deltaE_ciede2000(prediction_lab, target_lab)

        # Return mean Delta E as the loss
        return delta_e.mean()

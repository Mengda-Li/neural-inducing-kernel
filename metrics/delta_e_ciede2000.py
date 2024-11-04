# metrics/delta_e_ciede2000.py

import torch
from .metric import Metric

class DeltaECiede2000Metric(Metric):
    def __init__(self, kL=1, kC=1, kH=1):
        self.kL = kL
        self.kC = kC
        self.kH = kH

    def __call__(self, lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
        # Separate the L, a, b channels
        L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
        L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

        C1 = torch.sqrt(a1**2 + b1**2)
        C2 = torch.sqrt(a2**2 + b2**2)
        Cbar = 0.5 * (C1 + C2)
        G = 0.5 * (1 - torch.sqrt(Cbar**7 / (Cbar**7 + 25**7)))

        a1_prime, a2_prime = a1 * (1 + G), a2 * (1 + G)
        C1_prime, C2_prime = torch.sqrt(a1_prime**2 + b1**2), torch.sqrt(a2_prime**2 + b2**2)

        h1_prime = torch.atan2(b1, a1_prime)
        h2_prime = torch.atan2(b2, a2_prime)

        Lbar_prime = 0.5 * (L1 + L2)
        Cbar_prime = 0.5 * (C1_prime + C2_prime)
        hbar_prime = torch.where(
            (torch.abs(h1_prime - h2_prime) > torch.pi),
            (h1_prime + h2_prime + 2 * torch.pi) % (2 * torch.pi),
            (h1_prime + h2_prime) * 0.5,
        )

        T = (1 - 0.17 * torch.cos(hbar_prime - torch.deg2rad(30))
             + 0.24 * torch.cos(2 * hbar_prime)
             + 0.32 * torch.cos(3 * hbar_prime + torch.deg2rad(6))
             - 0.20 * torch.cos(4 * hbar_prime - torch.deg2rad(63)))

        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        delta_h_prime = h2_prime - h1_prime
        delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime / 2)

        SL = 1 + 0.015 * (Lbar_prime - 50)**2 / torch.sqrt(20 + (Lbar_prime - 50)**2)
        SC = 1 + 0.045 * Cbar_prime
        SH = 1 + 0.015 * Cbar_prime * T

        delta_theta = torch.deg2rad(30) * torch.exp(-((torch.rad2deg(hbar_prime) - 275) / 25)**2)
        Rc = 2 * torch.sqrt(Cbar_prime**7 / (Cbar_prime**7 + 25**7))

        RT = -torch.sin(2 * delta_theta) * Rc
        delta_E = torch.sqrt(
            (delta_L_prime / (SL * self.kL))**2 +
            (delta_C_prime / (SC * self.kC))**2 +
            (delta_H_prime / (SH * self.kH))**2 +
            RT * (delta_C_prime / (SC * self.kC)) * (delta_H_prime / (SH * self.kH))
        )
        return delta_E

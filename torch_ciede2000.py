import torch

def ciede2000(Lab1, Lab2):
    '''
    pytorch implementation of Delta E perceptual color difference, CIEDE2000
    '''

    # Constants for the CIEDE2000 formula
    kL = 1.0
    kC = 1.0
    kH = 1.0

    # Extract L, a, b values
    L1, a1, b1 = Lab1[:, 0], Lab1[:, 1], Lab1[:, 2]
    L2, a2, b2 = Lab2[:, 0], Lab2[:, 1], Lab2[:, 2]

    # Calculate C and h values
    C1 = torch.sqrt(a1**2 + b1**2)
    C2 = torch.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2

    G = 0.5 * (1 - torch.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    a1_prime = (1 + G) * a1
    a2_prime = (1 + G) * a2

    C1_prime = torch.sqrt(a1_prime**2 + b1**2)
    C2_prime = torch.sqrt(a2_prime**2 + b2**2)

    h1_prime = torch.atan2(b1, a1_prime)
    h1_prime = h1_prime + 2 * torch.pi * (h1_prime < 0).float()
    h2_prime = torch.atan2(b2, a2_prime)
    h2_prime = h2_prime + 2 * torch.pi * (h2_prime < 0).float()

    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    delta_h_prime = h2_prime - h1_prime
    delta_h_prime = delta_h_prime - 2 * torch.pi * (delta_h_prime > torch.pi).float()
    delta_h_prime = delta_h_prime + 2 * torch.pi * (delta_h_prime < -torch.pi).float()

    delta_H_prime = 2 * torch.sqrt(C1_prime * C2_prime) * torch.sin(delta_h_prime / 2)

    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2

    h_avg_prime = (h1_prime + h2_prime) / 2
    h_avg_prime = h_avg_prime - torch.pi * (torch.abs(h1_prime - h2_prime) > torch.pi).float()
    h_avg_prime = h_avg_prime + 2 * torch.pi * (h_avg_prime < 0).float()

    T = 1 - 0.17 * torch.cos(h_avg_prime - torch.deg2rad(torch.Tensor([30]))) + \
        0.24 * torch.cos(2 * h_avg_prime) + \
        0.32 * torch.cos(3 * h_avg_prime + torch.deg2rad(torch.Tensor([6]))) - \
        0.20 * torch.cos(4 * h_avg_prime - torch.deg2rad(torch.Tensor([63])))

    delta_theta = torch.deg2rad(torch.Tensor([30])) * torch.exp(-((h_avg_prime - torch.deg2rad(torch.Tensor([275]))) / torch.deg2rad(torch.Tensor([25])))**2)
    R_C = 2 * torch.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
    S_L = 1 + (0.015 * (L_avg_prime - 50)**2) / torch.sqrt(20 + (L_avg_prime - 50)**2)
    S_C = 1 + 0.045 * C_avg_prime
    S_H = 1 + 0.015 * C_avg_prime * T
    R_T = -torch.sin(2 * delta_theta) * R_C

    delta_E = torch.sqrt((delta_L_prime / (kL * S_L))**2 +
                         (delta_C_prime / (kC * S_C))**2 +
                         (delta_H_prime / (kH * S_H))**2 +
                         R_T * (delta_C_prime / (kC * S_C)) * (delta_H_prime / (kH * S_H)))

    return delta_E

if __name__ == '__main__':
    print("compare to pyciede2000 source implementation")

    import pyciede2000
    import numpy as np

    L1 = np.random.uniform(0, 100, 10)
    a1 = np.random.uniform(-128, 127, 10)
    b1 = np.random.uniform(-128, 127, 10)

    L2 = np.random.uniform(0, 100, 10)
    a2 = np.random.uniform(-128, 127, 10)
    b2 = np.random.uniform(-128, 127, 10)

    for (l_1,a_1,b_1), (l_2,a_2,b_2) in zip(zip(L1,a1,b1), zip(L2,a2,b2)):
        resnp = pyciede2000.ciede2000((l_1,a_1,b_1), (l_2,a_2,b_2))['delta_E_00']
        respt = ciede2000(torch.Tensor([[l_1,a_1,b_1]]), torch.Tensor([[l_2,a_2,b_2]]))
        print(f'pyciede implementation: {resnp:.5E}, pytorch implmentation: {respt.numpy()[0]:.5E}, difference: {np.abs(resnp-respt.numpy())[0]:.5E}')

    x1 = np.zeros((10,3))
    x1[:,0] = L1
    x1[:,1] = a1
    x1[:,2] = b1

    x2 = np.zeros((10,3))
    x2[:,0] = L2
    x2[:,1] = a2
    x2[:,2] = b2
    print(ciede2000(torch.Tensor(x1), torch.Tensor(x2)))

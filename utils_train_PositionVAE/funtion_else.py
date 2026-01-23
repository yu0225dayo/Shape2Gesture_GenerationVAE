import torch
import numpy as np

def z_rotation_matrix(angle_vec):
    """
    angle_vec: (B * sampling, 2) — (cosθ, sinθ)
    returns:   (B * sampling, 3, 3) z軸周りの回転行列
    """
    cos_theta = angle_vec[:, :, 0]
    sin_theta = angle_vec[:, :, 1]

    B, N, _ = angle_vec.shape 

    R = torch.zeros((B, N, 3, 3)).cuda()

    R[:, :, 0, 0] = cos_theta
    R[:, :, 0, 1] = -sin_theta
    R[:, :, 1, 0] = sin_theta
    R[:, :, 1, 1] = cos_theta
    R[:, :, 2, 2] = 1.0

    return R
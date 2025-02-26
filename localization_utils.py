import torch
from scipy.spatial.transform import Rotation as rotation


def normalize_translation(x, sf=1.0):
    # return torch.tan(x * (0.5 * torch.pi))
    return torch.tanh(x / sf)

def denormalize_translation(x, sf=1.0):
    # return torch.atan(x) / (0.5 * torch.pi)
    return sf * torch.atanh(x)

def normalize_angle(x):
    return x / (2 * torch.pi)

def denormalize_angle(x):
    return x * (2 * torch.pi)

def normalize_pose(x):
    x[:3] = normalize_translation(x[:3])
    x[3:] = normalize_angle(x[3:])
    return x

def denormalize_pose(x):
    x[:3] = denormalize_translation(x[:3])
    x[3:] = denormalize_angle(x[3:])
    return x

def batch_denormalize_pose(x):
    # not allowed to modify in-place for back prop
    # x[:, :3] = denormalize_translation(x[:, :3])
    # x[:, 3:] = denormalize_angle(x[:, 3:])
    # return x
    x_trans = denormalize_translation(x[:, :3])
    x_rot = denormalize_angle(x[:, 3:])
    return torch.cat([x_trans, x_rot], dim=1)

def convert_to_mtx(x):
    t = x[:3]
    r = x[3:]
    R_mat = rotation.from_euler('xyz', r, degrees=False).as_matrix()
    R_mat = torch.from_numpy(R_mat)
    P = torch.eye(4)
    P[:3, :3] = R_mat
    P[:3, 3] = t
    return P

def convert_to_vec(P):
    t = P[:3, 3]
    r = rotation.from_matrix(P[:3, :3]).as_euler('xyz')
    r = torch.tensor(r)
    pose_vec = torch.concat([t, r])
    return pose_vec


import torch
import numpy as np
from torchinfo import summary

from localization_model import *

def test_shapes():
    batch_size = 4
    img_channels = 2  # Stereo image pair (L and R as channels)
    img_height, img_width = 512, 512  # Example input resolution
    feature_dim = 128

    # Initialize model
    model = StereoPoseEstimation(input_size=(img_height, img_width))

    # Generate dummy input
    img1 = torch.randn(batch_size, img_channels, img_height, img_width)
    img2 = torch.randn(batch_size, img_channels, img_height, img_width)

    # Forward pass through feature extractor
    feat1 = model.feature_extractor(img1)
    feat2 = model.feature_extractor(img2)

    assert feat1.shape == (batch_size, feature_dim), f"Feature shape mismatch: {feat1.shape}"
    assert feat2.shape == (batch_size, feature_dim), f"Feature shape mismatch: {feat2.shape}"
    print("Feature extractor output shapes are correct.")

    # Forward pass through cross attention
    attended_feat = model.cross_attention(feat1, feat2)
    assert attended_feat.shape == (batch_size, feature_dim), f"Cross-attention shape mismatch: {attended_feat.shape}"
    print("Cross-attention output shape is correct.")

    # Forward pass through pose regressor
    pose = model.regressor(attended_feat)
    assert pose.shape == (batch_size, 6), f"Pose output shape mismatch: {pose.shape}"
    print("Pose regressor output shape is correct.")

    # Full model test
    pose_full = model(img1, img2)
    assert pose_full.shape == (batch_size, 6), f"Full model output shape mismatch: {pose_full.shape}"
    print("Full model output shape is correct.")

    model.eval()
    with torch.no_grad():
        summary(model, input_size=[(batch_size, img_channels, img_height, img_width), (batch_size, img_channels, img_height, img_width)])

def test_utils():
    p1 = [1,2,3, 
          np.deg2rad(10), np.deg2rad(20), np.deg2rad(40)]
    p2 = [2,3,3.1, 
          np.deg2rad(5), np.deg2rad(10), np.deg2rad(35)]

    p1 = torch.asarray(p1)
    p2 = torch.asarray(p2)

    print(f"p1: {p1}")
    print(f"p2: {p2}")

    p1_n = normalize_pose(p1)
    p2_n = normalize_pose(p2)

    print(f"p1 normalized: {p1_n}")
    print(f"p2 normalized: {p2_n}")

    print(f"p1 unorm: {denormalize_pose(p1_n)}")
    print(f"p2 unorm: {denormalize_pose(p2_n)}")

    p1_mat = convert_to_mtx(p1)
    p2_mat = convert_to_mtx(p2)

    print(f"p1 mat: {p1_mat}")
    print(f"p2 mat: {p2_mat}")

    relative_pose = torch.inverse(p1_mat) @ p2_mat
    
    print(f"relative pose mat: {relative_pose}")

    relative_pose = convert_to_vec(relative_pose)

    print(f"relative pose vec: {relative_pose}")

    relative_pose = normalize_pose(relative_pose)

    print(f"relative pose normalized: {relative_pose}")

# Run the test
test_shapes()
test_utils()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")
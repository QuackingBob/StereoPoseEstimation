import torch
import numpy as np
from torchinfo import summary
from pandas import DataFrame as df
import cv2

from localization_model import *
from data_loaders import *

def test_shapes():
    batch_size = 4
    img_channels = 2  # stereo pair L+R = 2
    img_height, img_width = 512, 512
    feature_dim = 128

    model = StereoPoseEstimation(input_size=(img_height, img_width))

    # dummy input
    img1 = torch.randn(batch_size, img_channels, img_height, img_width)
    img2 = torch.randn(batch_size, img_channels, img_height, img_width)

    # forward pass through feature extractor
    feat1 = model.feature_extractor(img1)
    feat2 = model.feature_extractor(img2)

    assert feat1.shape == (batch_size, feature_dim), f"Feature shape mismatch: {feat1.shape}"
    assert feat2.shape == (batch_size, feature_dim), f"Feature shape mismatch: {feat2.shape}"
    print("Feature extractor output shapes are correct.")

    # forward pass through cross attention
    attended_feat = model.cross_attention(feat1, feat2)
    assert attended_feat.shape == (batch_size, feature_dim), f"Cross-attention shape mismatch: {attended_feat.shape}"
    print("Cross-attention output shape is correct.")

    # forward pass through pose regressor
    pose = model.regressor(attended_feat)
    assert pose.shape == (batch_size, 6), f"Pose output shape mismatch: {pose.shape}"
    print("Pose regressor output shape is correct.")

    # full model test
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


def test_data_loader():
    data_path = "~/Documents/TrainingData/LAC/data"
    sample_name = "light"
    stereo_dir = f"stereo_pairs_{sample_name}"
    csv_filename = f"pose_{sample_name}.csv"

    img_path = os.path.expanduser(os.path.join(data_path, stereo_dir))
    csv_path = os.path.expanduser(os.path.join(data_path, csv_filename))

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = StereoDatasetEfficient(img_path, csv_path, transform, add_noise=False)

    df = pd.read_csv(csv_path)
    frame_indices = df["fname"].values  # ensure frames are accessed in order

    idx = np.random.randint(0, len(frame_indices) - 1)

    img_stk_t, img_stk_tp1, _ = dataset[20]

    im_l_1 = np.array(img_stk_t[0, :, :])
    im_r_1 = np.array(img_stk_t[1, :, :])
    im_l_2 = np.array(img_stk_tp1[0, :, :])
    im_r_2 = np.array(img_stk_tp1[1, :, :])

    pair_image = np.block([
        [im_l_1, im_r_1],
        [im_l_2, im_r_2]]
    )

    print(f"Stereo Image Shape: {im_l_1.shape}")
    
    cv2.imshow("Stereo Pair", pair_image)
    cv2.waitKey(5000)


# run the tests
test_shapes()
test_utils()
test_data_loader()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device: {device}")
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from localization_model import StereoPoseEstimation
from localization_utils import *
from data_loaders import StereoDataset  

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

dataset = StereoDataset(img_path, csv_path, transform, add_noise=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StereoPoseEstimation(input_size=(512, 512))
# model.load_state_dict(torch.load("model_wild-microwave-9_epoch_20.pth", map_location=device))
model.to(device)
model.eval()

# load true poses sequential order
df = pd.read_csv(csv_path)
frame_indices = df["fname"].values  # ensure frames are accessed in order
true_poses = [convert_to_mtx(torch.tensor(df.iloc[i, 1:].values, dtype=torch.float32)) for i in range(len(df))]

# initialize estimated path starting from the first true pose
estimated_poses = [true_poses[0]]

with torch.no_grad():
    for i in tqdm(range(len(frame_indices) - 1), desc="Evaluating model"):
        frame_t = frame_indices[i]
        frame_tp1 = frame_indices[i + 1]

        # load images at t and t + 1
        img_stk_t, img_stk_tp1, _ = dataset[i]  
        img_stk_t, img_stk_tp1 = img_stk_t.to(device).float(), img_stk_tp1.to(device).float()

        # relative pose
        pred_relative_pose = model(img_stk_t.unsqueeze(0), img_stk_tp1.unsqueeze(0)).squeeze(0).cpu()
        pred_relative_pose = denormalize_pose(pred_relative_pose)
        pred_relative_pose_mtx = convert_to_mtx(pred_relative_pose)

        # accumulate transformation
        new_pose = estimated_poses[-1] @ pred_relative_pose_mtx
        estimated_poses.append(new_pose)

# X, Y coordinates for plotting
true_x = [pose[0, 3] for pose in true_poses]
true_y = [pose[1, 3] for pose in true_poses]
est_x = [pose[0, 3] for pose in estimated_poses]
est_y = [pose[1, 3] for pose in estimated_poses]

# plot trajectories
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, label="True Path", color="blue", marker="o", markersize=3, linestyle="dashed")
plt.plot(est_x, est_y, label="Estimated Path", color="red", marker="x", markersize=3, linestyle="dashed")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("True vs Estimated Path")
plt.legend()
plt.grid()
plt.show()

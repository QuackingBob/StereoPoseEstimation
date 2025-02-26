import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb

from localization_model import *
from localization_utils import *
from data_loaders import StereoDatasetEfficient

def plot_trajectory(true_poses, estimated_poses, epoch, run_name):
    true_x = [pose[0, 3] for pose in true_poses]
    true_y = [pose[1, 3] for pose in true_poses]
    est_x = [pose[0, 3] for pose in estimated_poses]
    est_y = [pose[1, 3] for pose in estimated_poses]
    
    plt.figure(figsize=(8, 6))
    plt.plot(true_x, true_y, label="True Path", color="blue", marker="o", markersize=3, linestyle="dashed")
    plt.plot(est_x, est_y, label="Estimated Path", color="red", marker="x", markersize=3, linestyle="dashed")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"True vs Estimated Path (Epoch {epoch})")
    plt.legend()
    plt.grid()
    img_path = f"trajectory_{run_name}_epoch_{epoch}.png"
    plt.savefig(img_path)
    plt.close()
    return img_path

def train_model(model, dataset, batch_size, epochs, lr, device, normalized_pose=True):
    wandb.init(project="stereo_pose_estimation", config={
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr
    })

    name = str(wandb.run.name)
    print(f"Run Started {name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    model.train()
    true_poses = []
    estimated_poses = []
    
    df = pd.read_csv(dataset.csv_path)
    frame_indices = df["fname"].values  # ensure sequential order
    true_poses = [convert_to_mtx(torch.tensor(df.iloc[i, 1:].values, dtype=torch.float32)) for i in range(len(df))]
    estimated_poses.append(true_poses[0])
    
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for img_stk_1, img_stk_2, pose in pbar:
                img_stk_1 = img_stk_1.to(device).float()
                img_stk_2 = img_stk_2.to(device).float()
                pose = pose.to(device).float()

                optimizer.zero_grad()
                pred_pose = model(img_stk_1, img_stk_2)
                if not normalized_pose:
                    pred_pose = batch_denormalize_pose(pred_pose) 
                loss = loss_fn(pred_pose, pose)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=total_loss / len(dataloader))
                
                wandb.log({"batch_loss": loss.item()})
        
        epoch_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss": epoch_loss})

        if (epoch + 1) % 5 == 0:
            print("Performing Validation ...")
            for i in tqdm(range(len(frame_indices) - 1)):
                img_stk_t, img_stk_tp1, _ = dataset[i]  
                img_stk_t, img_stk_tp1 = img_stk_t.to(device).float(), img_stk_tp1.to(device).float()

                with torch.no_grad():
                    pred_relative_pose = model(img_stk_t.unsqueeze(0), img_stk_tp1.unsqueeze(0)).squeeze(0).cpu()
                    pred_relative_pose = denormalize_pose(pred_relative_pose)
                    pred_relative_pose_mtx = convert_to_mtx(pred_relative_pose)

                new_pose = estimated_poses[-1] @ pred_relative_pose_mtx
                estimated_poses.append(new_pose)
            
            img_path = plot_trajectory(true_poses, estimated_poses, epoch + 1, name)
            wandb.log({"trajectory_plot": wandb.Image(img_path)})
            print(f"Trajectory plot saved and uploaded at epoch {epoch + 1}")
        
        if (epoch + 1) % 10 == 0:
            model_path = f"model_{name}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)
            print(f"Model saved at epoch {epoch + 1}")
    
    wandb.finish()

if __name__ == "__main__":
    data_path = "~/Documents/TrainingData/LAC/data"
    sample_name = "light"

    stereo_dir = f"stereo_pairs_{sample_name}"
    csv_filename = f"pose_{sample_name}.csv"

    img_path = os.path.join(data_path, stereo_dir)
    csv_path = os.path.join(data_path, csv_filename)

    img_path = os.path.expanduser(img_path)
    csv_path = os.path.expanduser(csv_path)

    print(f"img_path: {img_path}")
    print(f"csv_path: {csv_path}")

    image_shape = (512, 512)

    transform = transforms.Compose([
        transforms.Resize(image_shape),
        transforms.ToTensor()
    ])
    dataset = StereoDatasetEfficient(img_path, csv_path, transform, add_noise=True, normalize=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = StereoPoseEstimation(input_size=image_shape)
    model.to(device)

    train_model(model, dataset, batch_size=32, epochs=20, lr=0.001, device=device, normalized_pose=False)

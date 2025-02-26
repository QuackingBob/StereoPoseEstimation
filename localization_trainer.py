import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image, ImageFilter
import pandas as pd
import os
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as rotation
import wandb

from localization_model import *
from localization_utils import *
from data_loaders import StereoDatasetEfficient


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
    for epoch in range(epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for img_stk_1, img_stk_2, pose in pbar:
                # note to self: float 32 so change here
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
                
                wandb.log({"batch_loss": loss.item()})  # log batch loss
        
        epoch_loss = total_loss / len(dataloader)
        wandb.log({"epoch_loss": epoch_loss})  # log epoch loss
        
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

    # resolve ~ path
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
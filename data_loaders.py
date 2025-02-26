import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import pandas as pd
import os
import random
import numpy as np
from localization_utils import convert_to_mtx, convert_to_vec, normalize_pose
from tqdm import tqdm

class StereoDatasetEfficient(Dataset):
    def __init__(self, image_folder, csv_path, transform=None, add_noise=False, normalize=True):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.add_noise = add_noise
        self.normalize_pose = normalize
        
        poses = torch.tensor(self.df.iloc[:, 1:].values, dtype=torch.float32)  # Load all at once
        self.relative_poses = self.precompute_relative_poses(poses)

        self.image_pairs = []

        print("Loading Image Paths")
        for i in tqdm(range(len(self.df) - 1)):
            self.image_pairs.append(
                (os.path.join(self.image_folder, f"{int(self.df.iloc[i]['fname'])}_L.png"),
                 os.path.join(self.image_folder, f"{int(self.df.iloc[i]['fname'])}_R.png"),
                 os.path.join(self.image_folder, f"{int(self.df.iloc[i+1]['fname'])}_L.png"),
                 os.path.join(self.image_folder, f"{int(self.df.iloc[i+1]['fname'])}_R.png"))
            )
    
    def __len__(self):
        return len(self.image_pairs)

    def precompute_relative_poses(self, poses):
        """ Precompute all relative poses before training """
        relative_poses = []
        print("Loading Pose Data")
        for i in tqdm(range(len(poses) - 1)):
            relative_pose = torch.inverse(convert_to_mtx(poses[i])) @ convert_to_mtx(poses[i + 1])
            if self.normalize_pose:
                relative_poses.append(normalize_pose(convert_to_vec(relative_pose)))
            else:
                relative_poses.append(convert_to_vec(relative_pose))
        return torch.stack(relative_poses)

    def load_image_pair(self, left_path, right_path):
        """ Load left and right images from file """
        left_img = Image.open(left_path).convert('L')
        right_img = Image.open(right_path).convert('L')
        return left_img, right_img

    def apply_augmentations(self, img):
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        return img
    
    def __getitem__(self, idx):
        left1_path, right1_path, left2_path, right2_path = self.image_pairs[idx]
        relative_pose = self.relative_poses[idx]  
        
        img1_L, img1_R = self.load_image_pair(left1_path, right1_path)
        img2_L, img2_R = self.load_image_pair(left2_path, right2_path)

        # Apply augmentations if needed
        if self.add_noise:
            img1_L, img1_R = self.apply_augmentations(img1_L), self.apply_augmentations(img1_R)
            img2_L, img2_R = self.apply_augmentations(img2_L), self.apply_augmentations(img2_R)

        # Convert to tensor and apply transformations
        if self.transform:
            img1_L, img1_R = self.transform(img1_L), self.transform(img1_R)
            img2_L, img2_R = self.transform(img2_L), self.transform(img2_R)

        # Stack images as [Left; Right] pairs
        img_stack_t = torch.cat([img1_L, img1_R], dim=0)
        img_stack_tp1 = torch.cat([img2_L, img2_R], dim=0)

        return img_stack_t, img_stack_tp1, relative_pose


class StereoDataset(Dataset):
    def __init__(self, image_folder, csv_path, transform=None, add_noise=False):
        self.image_folder = image_folder
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.add_noise = add_noise
    
    def __len__(self):
        return len(self.df) - 1   # use consecutive frames
    
    def load_image_pair(self, frame_num):
        left_img = Image.open(os.path.join(self.image_folder, f"{frame_num}_L.png")).convert('L')
        right_img = Image.open(os.path.join(self.image_folder, f"{frame_num}_R.png")).convert('L')
        return left_img, right_img
    
    def apply_augmentations(self, img):
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.3:
            img = img.point(lambda p: p * random.uniform(0.8, 1.2))
        return img
    
    def __getitem__(self, idx):
        frame1 = int(self.df.iloc[idx]['fname'])
        frame2 = int(self.df.iloc[idx + 1]['fname'])
        pose1 = torch.tensor(self.df.iloc[idx][1:].values, dtype=torch.float32)
        pose2 = torch.tensor(self.df.iloc[idx + 1][1:].values, dtype=torch.float32)
        
        img1_L, img1_R = self.load_image_pair(frame1)
        img2_L, img2_R = self.load_image_pair(frame2)
        
        if self.add_noise:
            img1_L, img1_R = self.apply_augmentations(img1_L), self.apply_augmentations(img1_R)
            img2_L, img2_R = self.apply_augmentations(img2_L), self.apply_augmentations(img2_R)
        
        if self.transform:
            img1_L, img1_R = self.transform(img1_L), self.transform(img1_R)
            img2_L, img2_R = self.transform(img2_L), self.transform(img2_R)
        
        img_stack_t = torch.cat([img1_L, img1_R], dim=0)
        img_stack_tp1 = torch.cat([img2_L, img2_R], dim=0) # tp1 for t+1
        
        # relative_pose = pose2 - pose1
        relative_pose = torch.inverse(convert_to_mtx(pose1)) @ convert_to_mtx(pose2)
        relative_pose = convert_to_vec(relative_pose)

        relative_pose = normalize_pose(relative_pose)
        
        return img_stack_t, img_stack_tp1, relative_pose
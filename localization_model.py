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

from localization_utils import *


class StereoFeatureExtractor(nn.Module):
    def __init__(self, channels, kernel_sizes, num_dense, dim_dense, input_size):
        super(StereoFeatureExtractor, self).__init__()

        # conv feature extractor
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=channels[i], 
                      out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i],
                      stride=1, padding=kernel_sizes[i] // 2)
            for i in range(len(kernel_sizes))
        ])

        self.max_pool = nn.MaxPool2d(2, 2)
        
        # finmd feature map size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, channels[0], *input_size)
            for conv in self.convs:
                dummy_input = conv(dummy_input)
                dummy_input = self.max_pool(dummy_input)
            _, c, h, w = dummy_input.shape  # final dimensions
            self.feature_map_size = (c, h, w)
        
        input_dim = c * h * w  # computed input dimension
        
        # MLP for feature embedding
        dense_layers = []
        for _ in range(num_dense):
            dense_layers.append(nn.Linear(input_dim, dim_dense))
            dense_layers.append(nn.ReLU())
            input_dim = dim_dense
        
        self.encoder_dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.max_pool(x)
        x = rearrange(x, 'b c h w -> b (c h w)')  # flatten 
        x = self.encoder_dense(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
    
    def forward(self, feat1, feat2):
        q = self.query(feat1)
        k = self.key(feat2)
        v = self.value(feat2)
        attn_weights = torch.softmax((q @ k.T) * self.scale, dim=-1)
        return attn_weights @ v

class PoseRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PoseRegressor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim), nn.Tanh()
        )
    
    def forward(self, x):
        return self.mlp(x)

class StereoPoseEstimation(nn.Module):
    def __init__(self, input_size=(64, 64)): 
        super(StereoPoseEstimation, self).__init__()
        self.feature_extractor = StereoFeatureExtractor(
            channels=[2, 16, 32, 64], 
            kernel_sizes=[5, 3, 3],
            num_dense=2,
            dim_dense=128,
            input_size=input_size  # pass input size for dynamic feature size computation
        )
        feature_dim = self.feature_extractor.feature_map_size[0] * \
                      self.feature_extractor.feature_map_size[1] * \
                      self.feature_extractor.feature_map_size[2]

        self.cross_attention = CrossAttention(dim=128)
        self.regressor = PoseRegressor(input_dim=128, output_dim=6)  # dx, dy, dz, dtheta, dphi, dgamma
    
    def forward(self, img1, img2):
        feat1 = self.feature_extractor(img1)
        feat2 = self.feature_extractor(img2)
        attended_feat = self.cross_attention(feat1, feat2)
        pose = self.regressor(attended_feat)
        return pose


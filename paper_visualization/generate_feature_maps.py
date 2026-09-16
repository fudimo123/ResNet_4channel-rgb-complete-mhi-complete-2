import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
# Path to the specific image
IMAGE_PATH = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub01\EP02_01f\img46.jpg"
OUTPUT_DIR = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization"

# --- 1. Load Model ---
# We use a standard ResNet18 pretrained on ImageNet
# This is sufficient to generate "feature maps" that look like feature maps
print("Loading ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# We only need the feature extractor part (up to layer4)
model = nn.Sequential(*list(model.children())[:-2]) # Remove avgpool and fc
model.eval()

# --- 2. Load and Preprocess Image ---
print(f"Loading image from {IMAGE_PATH}...")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    exit(1)

input_image = Image.open(IMAGE_PATH).convert('RGB')

# Preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # Add batch dimension

# --- 3. Extract Features ---
print("Extracting features...")
with torch.no_grad():
    feature_maps = model(input_batch) # Output shape: (1, 512, 7, 7)

print(f"Feature map shape: {feature_maps.shape}")
features = feature_maps[0] # (512, 7, 7)

# --- 4. Visualize ---

# A. Visualization 1: Mean Activation (Heatmap)
# Calculate the mean across all channels
mean_activation = torch.mean(features, dim=0).numpy()
# Normalize to 0-1 for visualization
mean_activation = (mean_activation - mean_activation.min()) / (mean_activation.max() - mean_activation.min())

plt.figure(figsize=(5, 5))
plt.imshow(mean_activation, cmap='viridis')
plt.axis('off')
plt.title("Mean Activation (Layer 4)")
plt.savefig(os.path.join(OUTPUT_DIR, "feature_map_mean.png"), bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved feature_map_mean.png")

# B. Visualization 2: Grid of Channels (The "Stack" Look)
# We'll plot the first 64 channels in an 8x8 grid
num_channels_to_show = 64
grid_size = 8 # 8x8

fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
# Remove spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for i in range(grid_size * grid_size):
    ax = axes[i // grid_size, i % grid_size]
    if i < features.shape[0]:
        channel_map = features[i].numpy()
        ax.imshow(channel_map, cmap='viridis') # viridis or gray
    ax.axis('off')

plt.savefig(os.path.join(OUTPUT_DIR, "feature_map_grid_64.png"), bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved feature_map_grid_64.png")

# C. Visualization 3: Top Active Channels
# Find the channels with the highest total activation
channel_activations = torch.sum(features, dim=(1, 2))
top_channels = torch.topk(channel_activations, 9).indices

fig, axes = plt.subplots(3, 3, figsize=(6, 6))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, channel_idx in enumerate(top_channels):
    ax = axes[i // 3, i % 3]
    channel_map = features[channel_idx].numpy()
    ax.imshow(channel_map, cmap='viridis')
    ax.axis('off')
    # ax.set_title(f"Ch {channel_idx}")

plt.savefig(os.path.join(OUTPUT_DIR, "feature_map_top9.png"), bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved feature_map_top9.png")

print("Done!")

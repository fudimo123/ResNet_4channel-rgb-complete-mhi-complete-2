import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Configuration ---
IMAGE_PATH = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\data\CASME2_RAW_selected\CASME2_RAW_selected\sub01\EP02_01f\img46.jpg"
OUTPUT_DIR = r"D:\Users\fuziyan\PycharmProjects\ResNet_4channel-rgb-complete-mhi-complete-2\paper_visualization"

# --- 1. Load Model ---
print("Loading ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-2])
model.eval()

# --- 2. Load and Preprocess Image ---
print(f"Loading image from {IMAGE_PATH}...")
if not os.path.exists(IMAGE_PATH):
    print(f"Error: Image not found at {IMAGE_PATH}")
    input_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
else:
    input_image = Image.open(IMAGE_PATH).convert('RGB')

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# --- 3. Extract Features ---
print("Extracting features...")
with torch.no_grad():
    feature_maps = model(input_batch)

features = feature_maps[0] # (512, 7, 7)

# --- 4. Simulate Channel Attention ---
# We want to multiply each channel by a weight between 0 and 1.
# Let's create some structured weights so some channels are clearly "attended to"
# and others are suppressed.
torch.manual_seed(42) # For reproducibility
channel_weights = torch.rand(512)
# Make some weights close to 1 and some close to 0 to emphasize the effect
channel_weights = torch.where(channel_weights > 0.5, channel_weights * 1.5, channel_weights * 0.2)
channel_weights = torch.clamp(channel_weights, 0.0, 1.0)

# Multiply features by channel weights
channel_refined_features = features * channel_weights.view(-1, 1, 1)

# --- 5. Visualize Channel-Refined Features ---
num_channels_to_show = 64
grid_size = 8

fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Calculate global min and max for the displayed channels to show intensity differences
global_max = channel_refined_features[:64].max().item()
global_min = channel_refined_features[:64].min().item()

for i in range(grid_size * grid_size):
    ax = axes[i // grid_size, i % grid_size]
    if i < channel_refined_features.shape[0]:
        channel_map = channel_refined_features[i].numpy()
        # Use vmin and vmax so that suppressed channels look darker
        ax.imshow(channel_map, cmap='viridis', vmin=global_min, vmax=global_max)
    ax.axis('off')

plt.savefig(os.path.join(OUTPUT_DIR, "channel_refined_feature_map.png"), bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved channel_refined_feature_map.png")

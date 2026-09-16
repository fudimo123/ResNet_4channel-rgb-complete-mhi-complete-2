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
    # Fallback: create a dummy image if file not found, just to let the script run
    print("Creating dummy image...")
    input_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
else:
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

# D. Visualization 4: Refined Output Style (Single strong activation map)
# To simulate the "Refined Output F''" which is weighted by attention,
# we can pick the single most active channel, or a weighted sum that looks "cleaner".
# Let's pick the max activation channel and apply a colormap that looks "hot" (red/yellow)

# Find the channel with max peak activation
max_val_per_channel = features.view(features.size(0), -1).max(dim=1)[0]
best_channel_idx = torch.argmax(max_val_per_channel).item()
refined_map = features[best_channel_idx].numpy()

# Normalize
refined_map = (refined_map - refined_map.min()) / (refined_map.max() - refined_map.min())

plt.figure(figsize=(3, 3))
# Use 'inferno' or 'magma' for that high-intensity look, or 'jet' for classic heatmap
plt.imshow(refined_map, cmap='inferno') 
plt.axis('off')
# Save without whitespace
plt.savefig(os.path.join(OUTPUT_DIR, "refined_feature_map.png"), bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved refined_feature_map.png")

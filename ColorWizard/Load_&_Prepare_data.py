# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 17:17:53 2024

@author: zefin


option of dataset:
    coco
    https://huggingface.co/datasets/nickpai/coco2017-colorization
    https://github.com/nick8592/Dataset-for-Image-Colorization/tree/main


    https://huggingface.co/datasets/csr/Image-Colorization
    
    
how to create a dataset
    https://huggingface.co/docs/datasets/upload_dataset
    
"""

from datasets import load_dataset

#1. Extract and Process the Image Data
import requests
from io import BytesIO
from PIL import Image
import numpy as np

def load_and_process_image(image_url):
    """
    Fetch the image from a URL, load it, and convert it to RGB format.
    """
    response = requests.get(image_url)
    response.raise_for_status()  # Ensure the request was successful
    image = Image.open(BytesIO(response.content)).convert("RGB")  # Load from URL
    return np.array(image)

#2. Resize Images
def resize_image(image, size=(256, 256)):
    """
    Resize a PIL image to the specified size.
    """
    if not isinstance(image, Image.Image):  # Check if image is a PIL Image
        image = Image.fromarray(image)      # Convert NumPy array to PIL Image
    return np.array(image.resize(size, Image.BICUBIC))


#3. Generate Grayscale Inputs
def rgb_to_grayscale(image):
    return np.array(Image.fromarray(image).convert("L"))

#4. Normalize Pixel Values
def normalize_image(image):
    return image / 255.0  # Normalize to [0, 1]

# Pair Grayscale and RGB Images
def preprocess_sample(image_url, size=(256, 256)):
    image = load_and_process_image(image_url)
    resized_image = resize_image(image, size)
    grayscale_image = rgb_to_grayscale(resized_image)
    normalized_grayscale = normalize_image(grayscale_image)
    normalized_color = normalize_image(resized_image)
    return normalized_grayscale, normalized_color

#%%
#7. Create Preprocessed Dataset
from datasets import DatasetDict

def preprocess_function(example):
    grayscale, color = preprocess_sample(example['coco_url'])  # Adjust based on image location.
    example['grayscale'] = grayscale
    example['color'] = color
    return example

dataset= load_dataset("nickpai/coco2017-colorization", split="validation",revision="main")

preprocessed_dataset = dataset.map(preprocess_function, batched=False)


#%% visualization

example = preprocessed_dataset[0]  # First sample
grayscale_image = example['grayscale']
color_image = example['color']

# Plotting
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title('Grayscale')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(color_image)
plt.title('Color')
plt.axis('off')
plt.tight_layout()
plt.show()


#%%
#8. Prepare for Model Training
import torch

# Function to convert images to PyTorch tensors
def to_tensor(image):
    return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # For RGB images (C, H, W)

def preprocess_for_torch(example):
    """
    Convert grayscale and color images to PyTorch tensors.
    """
    grayscale_tensor = torch.tensor(example['grayscale'], dtype=torch.float32).unsqueeze(0)  # (1, H, W)
    color_tensor = to_tensor(example['color'])  # (3, H, W)
    return {'input': grayscale_tensor, 'target': color_tensor}

# Apply the preprocessing function
torch_ready_dataset = preprocessed_dataset.map(preprocess_for_torch, batched=False)

# Set the format to PyTorch for easy batch loading
torch_ready_dataset.set_format(type="torch", columns=["input", "target"])


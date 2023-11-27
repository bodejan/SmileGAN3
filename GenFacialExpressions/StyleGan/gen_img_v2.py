import sys
import torch
import requests
import pickle
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image

def gen_stylegan3_img_v2(name, download = False):
    # Check if MPS (Metal) GPU is available and use it
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #print(f"Using device: {device}")

    model_path = '/Users/janbode/computer-vision/GenFacialExpressions/StyleGan/stylegan3-r-ffhq-1024x1024.pkl'
    if download:
        # Download the pre-trained StyleGAN3 FFHQ model
        model_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl'
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    # Load the model on MPS or CPU
    with open(model_path, 'rb') as f:
        model = pickle.load(f)['G_ema'].to(device)

    z = torch.randn([1, model.z_dim], device=device)  # Latent vectors
    img = model(z, None)

    # Normalize and convert the images for saving
    normalized_img = (img + 1) / 2
    normalized_img = normalized_img.clamp(0, 1)

    # Convert the tensor to a PIL Image
    # Assuming the image tensor is in the shape [1, C, H, W] and in the range [0, 1]
    img_pil = transforms.ToPILImage()(normalized_img.squeeze(0))

    # Save the image
    image_path = f'/Users/janbode/computer-vision/GenFacialExpressions/StyleGan/img/{name}.png'
    img_pil.save(image_path)

    time.sleep(3)

    return name, z, img_pil

#gen_stylegan3_img_v2('test', True)
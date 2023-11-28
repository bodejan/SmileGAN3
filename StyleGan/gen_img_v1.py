import sys
import torch
import requests
import pickle
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import time
import os
from PIL import Image

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
# Add the StyleGAN3 folder to Python path
sys.path.append('./stylegan3')

def gen_stylegan3_imgs(num_images, download = False):
    start = time.time()
    # Check if MPS (Metal) GPU is available and use it
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = 'stylegan3-r-ffhq-1024x1024.pkl'
    if download:
        # Download the pre-trained StyleGAN3 FFHQ model
        model_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl'
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    # Load the model on MPS or CPU
    with open(model_path, 'rb') as f:
        model = pickle.load(f)['G_ema'].to(device)

    for i in range(1, num_images + 1):
        i_offset = 90

        z = torch.randn([1, model.z_dim], device=device)  # Latent vectors
        img = model(z, None)

        # Normalize and convert the images for saving
        normalized_img = (img + 1) / 2
        normalized_img = normalized_img.clamp(0, 1)

        # Save images
        image_folder = './img'  # Folder where images will be saved
        single_img_pil = Image.fromarray(normalized_img[0].mul(255).permute(1, 2, 0).byte().cpu().numpy())
        single_img_pil.save(f'{image_folder}/image_{i + i_offset}.png')  # Save the image

        print(f'Generated {i} out of {num_images} images in {time.time() - start}s.')

        time.sleep(3)


gen_stylegan3_imgs(1)


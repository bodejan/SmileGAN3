import pandas as pd
import requests
import torch
import pickle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

import sys
import time
import os

from model import FEAR, DISGUST, ANGRY, SURPRISE, SAD, HAPPY, NEUTRAL
from vector import add_and_multiply_vectors

# DANGEROUS, may lead to crash
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def gen_emotion_row(name, random_factor=0.25, export=False, visualize=True):

    random = torch.randn(512)
    random_img = gen_img_for_vector(random)
    if export:
        os.makedirs('img/result/random', exist_ok=True)
        random_img.save(f'img/result/random/random_{name}.png')
    
    neutral_base = add_and_multiply_vectors(random, NEUTRAL, 0.2)

    emotions = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Surprise', 'Sad']
    emotion_factors = [neutral_base, HAPPY, ANGRY, DISGUST, FEAR, SURPRISE, SAD]
    images = []

    for i, emotion_factor in enumerate(emotion_factors):
        if emotions[i] == 'Neutral':
            emotion_input = emotion_factors[i]
        else:
            emotion_input = add_and_multiply_vectors(random, emotion_factor, random_factor)
        emotion_img = gen_img_for_vector(emotion_input)
        if export:
            os.makedirs(f'img/result/{emotions[i].lower()}', exist_ok=True)
            emotion_img.save(f'img/result/{emotions[i].lower()}/{emotions[i].lower()}_{name}.png')
        images.append(emotion_img)

    
    # Plotting
    fig, axs = plt.subplots(1, len(images) + 1, figsize=(20, 5))
    axs[0].imshow(random_img)
    axs[0].set_title(f'Random')
    axs[0].axis('off')
    for i, img in enumerate(images):
        axs[i + 1].imshow(img)
        axs[i + 1].set_title(emotions[i])
        axs[i + 1].axis('off')

    if visualize:
        plt.show()

    if export:
        plt.savefig(f'img/result/row/row_{name}.png', bbox_inches='tight')
        plt.close(fig)

def show_random_img(download = False):
    device = select_gpu_and_download(download)

    # Load the model on MPS or CPU
    with open('stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
        model = pickle.load(f)['G_ema'].to(device)

    # Generate random vector
    z = torch.randn([1, model.z_dim], device=device)  # Latent vectors

    # Generate the image
    img = model(z, None)

    # Normalize and convert the images for visualization
    normalized_img = (img + 1) / 2
    normalized_img = normalized_img.clamp(0, 1)

    # Convert the tensor
    img_pil = transforms.ToPILImage()(normalized_img.squeeze(0))
    plt.imshow(img_pil)
    plt.title('Random image')
    plt.show()


def gen_img_with_vector(name, download = False):
    device = select_gpu_and_download(download)

    # Load the model on MPS or CPU
    with open('stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
        model = pickle.load(f)['G_ema'].to(device)

    z = torch.randn([1, model.z_dim], device=device)  # Latent vectors
    img = model(z, None)

    # Normalize and convert the images for saving
    normalized_img = (img + 1) / 2
    normalized_img = normalized_img.clamp(0, 1)

    # Convert the tensor to a PIL Image
    img_pil = transforms.ToPILImage()(normalized_img.squeeze(0))

    # Save the image
    image_path = f'img/train/train_img/{name}.png'
    img_pil.save(image_path)

    time.sleep(3)

    return z


def gen_img_for_vector(vector, download=False):
    device = select_gpu_and_download(download)

    # Load the model on MPS or CPU
    with open('stylegan3-r-ffhq-1024x1024.pkl', 'rb') as f:
        model = pickle.load(f)['G_ema'].to(device)

    # Use the provided vector
    z = torch.tensor(vector, device=device).unsqueeze(0)  # Ensure z is a 2D tensor

    # Generate the image
    img = model(z, None)

    # Normalize and convert the images for visualization
    normalized_img = (img + 1) / 2
    normalized_img = normalized_img.clamp(0, 1)

    # Convert the tensor
    img_pil = transforms.ToPILImage()(normalized_img.squeeze(0))

    return img_pil

def gen_img_with_vector_train_data(num_imgs, offset = 0):
    for i in range(1, num_imgs+1):
        name = i + offset
        z = gen_img_with_vector(str(name), False)
        label = ''

        z = z.cpu().detach().numpy().flatten().tolist()

        save_to_two_csvs(name, z, label)


# ---------------------------------------------- Utils ---------------------------------------------- #

def select_gpu_and_download(download):
    # Check if MPS (Metal) GPU is available and use it
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    #print(f"Using device: {device}")

    model_path = 'stylegan3-r-ffhq-1024x1024.pkl'
    if download:
        print('Downloading stylegan3-r-ffhq-1024x1024.pkl, this might take some time...')
        # Download the pre-trained StyleGAN3 FFHQ model
        model_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhq-1024x1024.pkl'
        
        response = requests.get(model_url)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    
    return device

def save_to_two_csvs(name, z, label):
    label_filename = 'img/train/labels.csv'
    vector_filename = 'img/train/vectors.csv'
    
    # Create a DataFrame for labels
    label_df = pd.DataFrame({'name': [name], 'label': [label]})
    # Create a DataFrame for vectors
    vector_df = pd.DataFrame({'name': [name], 'z_values': [z]})
    
    # Save label DataFrame to CSV
    if os.path.exists(label_filename):
        # File exists, append the data without the header
        label_df.to_csv(label_filename, mode='a', header=False, index=False)
    else:
        # File does not exist, create and write the data with the header
        label_df.to_csv(label_filename, header=True, index=False)
    
    # Save vector DataFrame to CSV
    if os.path.exists(vector_filename):
        # File exists, append the data without the header
        vector_df.to_csv(vector_filename, mode='a', header=False, index=False)
    else:
        # File does not exist, create and write the data with the header
        vector_df.to_csv(vector_filename, header=True, index=False)
import sys
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch

from Classifier.classifier import classify_emotion
from StyleGan.gen_img_v2 import gen_stylegan3_img_v2
from StyleGan.gen_img_v3 import gen_stylegan3_img_v3
from model import FEAR, DISGUST, ANGRY, SURPRISE, SAD, HAPPY, NEUTRAL


# Add the StyleGAN3 folder to Python path
sys.path.append('stylegan3')
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

def main(num_imgs, offset = 0):
    for i in range(1, num_imgs+1):
        name = i + offset
        _, z, img_pil = gen_stylegan3_img_v2(str(name), False)
        #label = classify_emotion(img_pil)
        # Classify label using GPT-4
        label = ''

        z = z.cpu().detach().numpy().flatten().tolist()

        save_to_two_csvs(name, z, label)

        print(f'Generated {i} out of {num_imgs} images and vectors.')


def save_to_two_csvs(name, z, label):
    label_filename = './labels.csv'
    vector_filename = './vectors.csv'
    
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

#main(100, 900)

def gen_avg_vectors():
    angry = gen_stylegan3_img_v3(ANGRY, 'ANGRY')
    disgust = gen_stylegan3_img_v3(DISGUST, 'DISGUST')
    fear = gen_stylegan3_img_v3(FEAR, 'FEAR')
    surprise = gen_stylegan3_img_v3(SURPRISE, 'SURPRISE')
    sad = gen_stylegan3_img_v3(SAD, 'SAD')
    neutral = gen_stylegan3_img_v3(NEUTRAL, 'NEUTRAL')
    happy = gen_stylegan3_img_v3(HAPPY, 'HAPPY')

    # Plotting
    fig, axs = plt.subplots(1, 7, figsize=(20, 5))
    axs[0].imshow(angry)
    axs[0].set_title(f'Angry avg. vector.')
    axs[0].axis('off')

    axs[1].imshow(disgust)
    axs[1].set_title(f'Disgust avg. vector.')
    axs[1].axis('off')

    axs[2].imshow(fear)
    axs[2].set_title(f'Fear avg. vector.')
    axs[2].axis('off')

    axs[3].imshow(surprise)
    axs[3].set_title(f'Surprise avg. vector.')
    axs[3].axis('off')

    axs[4].imshow(sad)
    axs[4].set_title(f'Sad avg. vector.')
    axs[4].axis('off')

    axs[5].imshow(neutral)
    axs[5].set_title(f'Neutral avg. vector.')
    axs[5].axis('off')

    axs[6].imshow(happy)
    axs[6].set_title(f'Happy avg. vector.')
    axs[6].axis('off')

    plt.savefig('avg_vectors', bbox_inches='tight')
    plt.close(fig)

#gen_avg_vectors()

def gen_mix(plt_name, factor=0.75):
    def add_and_multiply_lists(list_a, list_b, factor):
        return [a * (1-factor) + b * factor for a, b in zip(list_a, list_b)]
    
    random = torch.randn(512)
    random_img = gen_stylegan3_img_v3(random, f'Random, factor: {factor}')
    random_img.save(f'img/result/random/random_{plt_name}.png')

    neutral_base = add_and_multiply_lists(random, NEUTRAL, factor)

    emotions = ['Neutral', 'Happy', 'Angry', 'Disgust', 'Fear', 'Surprise', 'Sad']
    emotion_factors = [neutral_base, HAPPY, ANGRY, DISGUST, FEAR, SURPRISE, SAD]
    images = []

    for i, emotion_factor in enumerate(emotion_factors):
        if emotions[i] == 'Neutral':
            emotion_input = emotion_factors[i]
        else:
            emotion_input = add_and_multiply_lists(random, emotion_factor, factor)
        emotion_img = gen_stylegan3_img_v3(emotion_input, f'{emotions[i]}, factor: {factor}')
        # save img
        emotion_img.save(f'img/result/{emotions[i].lower()}/{emotions[i].lower()}_{plt_name}.png')
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

    #plt.show()
    plt.savefig(f'img/result/row/row_{plt_name}.png', bbox_inches='tight')
    plt.close(fig)


for i in range(0, 50):
    gen_mix(str(i))


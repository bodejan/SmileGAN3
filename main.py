import sys
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch

from stylegan3 import show_random_img, gen_img_with_vector, gen_img_for_vector, gen_img_with_vector_train_data
from model import FEAR, DISGUST, ANGRY, SURPRISE, SAD, HAPPY, NEUTRAL


# Add the StyleGAN3 folder to Python path
sys.path.append('stylegan3')
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

#main(100, 900)

def gen_avg_vectors():
    angry = gen_img_for_vector(ANGRY)
    disgust = gen_img_for_vector(DISGUST)
    fear = gen_img_for_vector(FEAR)
    surprise = gen_img_for_vector(SURPRISE)
    sad = gen_img_for_vector(SAD)
    neutral = gen_img_for_vector(NEUTRAL)
    happy = gen_img_for_vector(HAPPY)

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




#for i in range(50, 100):
#    gen_mix(str(i))

#show_random_img(True)

#gen_img_with_vector_train_data(2, 1000)


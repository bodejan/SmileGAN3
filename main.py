import sys
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch

from stylegan3 import show_random_img, gen_img_with_vector, gen_img_for_vector, gen_img_with_vector_train_data, gen_emotion_row
from model import FEAR, DISGUST, ANGRY, SURPRISE, SAD, HAPPY, NEUTRAL
from visualize import visualize_avg_vectors


# Add the StyleGAN3 folder to Python path
sys.path.append('stylegan3')
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'



#for i in range(50, 100):
#    gen_mix(str(i))

#show_random_img(True)

#gen_img_with_vector_train_data(2, 1000)

#gen_emotion_row('Test')

visualize_avg_vectors(True)
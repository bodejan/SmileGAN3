
import time
import matplotlib.pyplot as plt

from stylegan3 import gen_img_for_vector
from model import FEAR, DISGUST, ANGRY, SURPRISE, SAD, HAPPY, NEUTRAL

def visualize_avg_vectors(export = False):
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
    axs[0].set_title(f'Angry')
    axs[0].axis('off')

    axs[1].imshow(disgust)
    axs[1].set_title(f'Disgust')
    axs[1].axis('off')

    axs[2].imshow(fear)
    axs[2].set_title(f'Fear')
    axs[2].axis('off')

    axs[3].imshow(surprise)
    axs[3].set_title(f'Surprise')
    axs[3].axis('off')

    axs[4].imshow(sad)
    axs[4].set_title(f'Sad')
    axs[4].axis('off')

    axs[5].imshow(neutral)
    axs[5].set_title(f'Neutral')
    axs[5].axis('off')

    axs[6].imshow(happy)
    axs[6].set_title(f'Happy')
    axs[6].axis('off')

    if export:
        plt.savefig('img/train/avg_vectors.png', bbox_inches='tight')

    plt.show()

    if export:
        time.sleep(5)
        plt.close(fig)

import time
import matplotlib.pyplot as plt
from PIL import Image
import os

from smilegan3 import gen_img_for_vector
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

def create_image_grid(folder_path, output_path = None, grid_size=(10, 10), image_size=(100, 100)):
    images = []

    # Load 100 images from the folder
    for filename in os.listdir(folder_path)[:100]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder_path, filename))
            img = img.resize(image_size)  # Resize if necessary
            images.append(img)

    # Create a new image with the appropriate size
    grid_img = Image.new('RGB', (image_size[0] * grid_size[0], image_size[1] * grid_size[1]))

    # Paste the images into the grid
    for i, img in enumerate(images):
        grid_x = i % grid_size[0] * image_size[0]
        grid_y = i // grid_size[0] * image_size[1]
        grid_img.paste(img, (grid_x, grid_y))
    
    grid_img.show()
    if output_path:
        grid_img.save(output_path)

    return grid_img

def plot_all_emotions():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'random']
    for emotion in emotions:
        create_image_grid(f'img/result/{emotion}/', f'img/result/{emotion}/{emotion}_faces.png')

    create_image_grid(f'img/result/row/', f'img/result/row/all_faces.png', (4, 25), (800, 100))

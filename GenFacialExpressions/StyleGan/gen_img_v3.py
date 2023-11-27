import requests
import torch
import pickle
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def gen_stylegan3_img_v3(vector, name='generated_image', download=False):
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

    # Use the provided vector
    z = torch.tensor(vector, device=device).unsqueeze(0)  # Ensure z is a 2D tensor

    # Generate the image
    img = model(z, None)

    # Normalize and convert the images for visualization
    normalized_img = (img + 1) / 2
    normalized_img = normalized_img.clamp(0, 1)

    # Convert the tensor
    img_pil = transforms.ToPILImage()(normalized_img.squeeze(0))
    #plt.imshow(img_pil)
    #plt.title(name)
    #plt.show()

    return img_pil

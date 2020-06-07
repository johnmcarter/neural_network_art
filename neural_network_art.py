'''
Follows tutorial from 
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size - use bigger size if GPU is available
IMG_SIZE = 512 if torch.cuda.is_available() else 128

# Scale the image and make it a tensor
loader = transforms.Compose([transforms.Resize(IMG_SIZE),
                            transforms.ToTensor()])

def load_image(name):
    im = Image.open(name)
    im = loader(im).unsqueeze(0)
    
    return im.to(device, torch.float)

style = load_image("images/crucifixion.jpg")
content = load_image("images/plaza_de_espana.jpg")

# Check to make sure content and style are the same size. Exit if they aren't
if style.size() != content.size():
    print("Style and content photos must be same size!")
    exit()

plt.ion()

def show_img(tensor, name=None):
    '''
    Displays the image (tensor) passed in for 1 second
    '''
    im = tensor.cpu().clone()
    im = transforms.ToPILImage()(im.squeeze(0))
    plt.imshow(im)
    if name:
        plt.title(name)
    plt.pause(1)

plt.figure()
show_img(style, name="Style image")

plt.figure()
show_img(content, name="Content image")
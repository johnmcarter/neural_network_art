'''
Follows tutorial from 
https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
'''

from PIL import Image
from copy import deepcopy
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

# Style image from https://www.pablopicasso.org/crucifixion.jsp
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

# import the pre-trained CNN
cnn = models.vgg19(pretrained=True).features.to(device).eval()

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def compute_gram(input):
    dim_1, dim_2, dim_3, dim_4 = input.size()
    feat_map = input.reshape(dim_1 * dim_2, dim_3 * dim_4)

    # Compute gram matrix as dot product of feat_map and feat_map transposed
    gram_matrix = torch.mm(feat_map, feat_map.t())
    #Normalize gram matrix by dividing by dimension sizes
    gram_matrix = gram_matrix.div(dim_1 * dim_2 * dim_3 * dim_4)

    return gram_matrix

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # Make dimensions of mean and std compatible, so -1 for the number of channels
        # in the image and 1's for the height and width of the image
        self.mean = torch.tensor(mean).reshape(-1, 1, 1)
        self.std = torch.tensor(std).reshape(-1, 1, 1)

    def forward(self, img):
        #Normalize image
        return (img - self.mean)/self.std

class ContentMse(nn.Module):
    def __init__(self, target,):
        super(ContentMse, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleMse(nn.Module):
    def __init__(self, target_feat):
        super(StyleMse, self).__init__()
        self.target = compute_gram(target_feat).detach()
    
    def forward(self, input):
        gram_matrix = compute_gram(input)
        self.loss = F.mse_loss(gram_matrix, target)
        return input

def get_loss_modules(cnn, style_img, content_img, mean, std, 
                    content_layers = content_layers, style_layers = style_layers):
    
    cnn = deepcopy(cnn)
    content_loss, style_loss = [], []

    normalized = Normalize(mean, std).to(device)

    model = nn.Sequential(normalized)

    print(model)

get_loss_modules(cnn, style, content, mean, std)
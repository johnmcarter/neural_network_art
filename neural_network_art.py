'''
Portions of code adapted from
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
from torchvision.utils import save_image

import argparse

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size - use bigger size if GPU is available
IMG_SIZE = 512 if torch.cuda.is_available() else 128

# Scale the image and make it a tensor
loader = transforms.Compose([transforms.Resize(IMG_SIZE),
                            transforms.ToTensor()])

def load_image(name):
    '''
    Load an image to transform it into a tensor and return it
    @param name: image name to open
    '''
    im = Image.open(name)
    im = loader(im).unsqueeze(0)
    
    return im.to(device, torch.float)

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

def compute_sigma(input):
    '''
    Compute the sigma matrix by multiplying the feature map and
    and the transposed feature map
    '''
    dim_1, dim_2, dim_3, dim_4 = input.size()
    feat_map = input.reshape(dim_1 * dim_2, dim_3 * dim_4)

    # Compute sigma as dot product of feat_map and feat_map transposed
    sigma = torch.mm(feat_map, feat_map.t())
    # Normalize sigma by dividing by dimension sizes
    sigma = sigma.div(dim_1 * dim_2 * dim_3 * dim_4)

    return sigma

class Normalize(nn.Module):
    '''
    Normalize the model by subtracting the mean and dividing by
    the standard deviation
    '''
    def __init__(self, mean, std):
        '''
        @param mean: means for each dimension
        @param std: std's for each dimension
        '''
        super(Normalize, self).__init__()
        # Make dimensions of mean and std compatible, so -1 for the number of channels
        # in the image and 1's for the height and width of the image
        self.mean = torch.as_tensor(mean).reshape(-1, 1, 1)
        self.std = torch.as_tensor(std).reshape(-1, 1, 1)

    def forward(self, img):
        '''
        Normalize param 'img'
        '''
        return (img - self.mean)/self.std

class ContentMSE(nn.Module):
    '''
    Get the mean squared error loss for the content image
    '''
    def __init__(self, target,):
        super(ContentMSE, self).__init__()
        self.target = target.detach()
    
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleMSE(nn.Module):
    '''
    Get the mean squared error loss for the style painting
    '''
    def __init__(self, target_feat):
        super(StyleMSE, self).__init__()
        self.target = compute_sigma(target_feat).detach()
    
    def forward(self, input):
        '''
        Compute the sigma matrix, and then compute the loss between it
        and the target using mean squared error
        '''
        sigma = compute_sigma(input)
        self.loss = F.mse_loss(sigma, self.target)
        return input

def get_model_and_losses(cnn, style, content, mean, std, 
                    content_layers, style_layers):
    '''
    Get the CNN and the losses for the style and content images to be used
    in the run() function.
    @param cnn: the imported pre-trained CNN
    @param style: input style painting image
    @param content: input content image
    @param mean: initial means for each dimension
    @param std: initial std's for each dimension
    @param content_layers: depth layers to compute content losses
    @param style_layers: depth layers to compute style losses

    @return model: the CNN model we made
    @return style_losses: list of losses for style painting image
    @return content_losses: list of losses for content image
    '''
    
    cnn = deepcopy(cnn)
    content_losses, style_losses = [], []

    normalized = Normalize(mean, std).to(device)
    model = nn.Sequential(normalized)

    conv_count = 0

    for layer in cnn.children():
        # Check if it's a convolution layer
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            layer_name = 'conv_{}'.format(conv_count)
        # Check if it's a pooling layer
        elif isinstance(layer, nn.MaxPool2d):
            layer_name = 'pool_{}'.format(conv_count)
        # Check if it's an activation layer
        elif isinstance(layer, nn.ReLU):
            layer_name = 'relu_{}'.format(conv_count)
            layer = nn.ReLU(inplace=False)
        # Check if it's a batch normalization layer
        elif isinstance(layer, nn.BatchNorm2d):
            layer_name = 'bn_{}'.format(conv_count)
        else:
            print('\033[91mUnexpected layer: {} \033[0m'\
                                .format(layer.__class__.__name__))

        model.add_module(layer_name, layer)

        # Check if we're in one of the style layers
        if layer_name in style_layers:
            target_feat = model(style).detach()
            loss = StyleMSE(target_feat)
            model.add_module("style_loss_{}".format(conv_count), loss)
            style_losses.append(loss)

        # Check if we're in one of the content layers
        if layer_name in content_layers:
            target_feat = model(content).detach()
            loss = ContentMSE(target_feat)
            model.add_module("style_loss_{}".format(conv_count), loss)
            content_losses.append(loss)

    # Remove layers after the last style and content losses
    current_layer = len(model)-1
    while not isinstance(model[current_layer], StyleMSE) and \
        not isinstance(model[current_layer], ContentMSE) and (current_layer >= 0):
        current_layer -= 1

    model = model[:(current_layer+1)]

    return model, style_losses, content_losses


def run(cnn, mean, std, content, style, image, \
        content_layers, style_layers, style_weight, content_weight, epochs=300):
    '''
    Perform the style transfer. Run epochs times and build the mixture
    of content and style according to the style_weight, and content_weight.
    @param cnn: the imported pre-trained CNN
    @param mean: initial means for each dimension
    @param std: initial std's for each dimension
    @param content: input content image
    @param style: input style painting image
    @param input_image: initially a clone of content
    @param epochs: number of times to run
    @param style_weight: amount of influence to give style painting relative
    to the content
    @param content_weight: amount of influence to give content image relative
    to the style

    @return image: the final image
    '''
    print("\u001b[1m\033[92mStarting the content/style transfer...\033[0m")
    # Get the model and losses
    model, style_losses, content_losses = \
        get_model_and_losses(cnn, style, content, mean, std, \
                                content_layers, style_layers)

    # Optimize with LBFGS
    optimizer = optim.LBFGS([image.requires_grad_()])

    run = [0]
    while run[0] < epochs:
        # Define as closure() because it's required by LBFGS
        def closure():
            s_score = 0
            c_score = 0
            image.data.clamp_(0,1)
            optimizer.zero_grad()
            model(image)

            # Populate all the losses from both content and style
            for loss in content_losses:
                c_score += loss.loss
            for loss in style_losses:
                s_score += loss.loss

            # Scale the scores by the weights passed into the function and total
            # and then to backprop
            total_loss = c_score*content_weight + s_score*style_weight
            total_loss.backward(retain_graph=True)

            run[0] += 1

            # Print current stats every 20 iterations
            if run[0] % 20 == 0:
                print("\u001b[35mrun iteration {}".format(run))
                print("\u001b[33mstyle loss: {:4f}".format(s_score.item()))
                print("content loss: {:4f}".format(c_score.item()))
                print("\u001b[0m")

            return c_score + s_score
            
        optimizer.step(closure)

    # Make the output image have values between 0 and 1 - inplace
    image.data.clamp_(0,1)

    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Blends a content image and style painting')
    parser.add_argument('style', type=str, help='Style image')
    parser.add_argument('content', type=str, help='Content image')
    parser.add_argument('--sw', type=float, default=1,
                        help='Style weight (float between 0 and 1)')
    parser.add_argument('--cw', type=float, default=1e-5,
                        help='Content weight (float between 0 and 1)')
    parser.add_argument('--output', type=str, default='images/output.png',
                        help='Place to save the output')
    args = parser.parse_args()

    style = load_image(args.style)
    content = load_image(args.content)
    input_image = content.clone()

    # Check to make sure content and style are the same size. 
    # If they aren't, make style size match content size.
    if style.size() != content.size():
        print("\u001b[31mStyle and content photos are not the same size.")
        print("Style image will be scaled to size of content image.\u001b[0m")
        style = F.interpolate(style, size=(content.shape[2:]), 
                                mode='bilinear', align_corners=False)

    plt.ion()

    # Import the pre-trained CNN from torchvision
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # Define the style and content layers we'll use in our CNN
    # The style image will be passed through 5 layers, while the content
    # image will only need to be passed through 1 layer
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # Set initial mean and standard deviation based on what the VGG-19
    # model was trained on
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    print("\u001b[35m")
    print("Style image:", args.style)
    print("Content image:", args.content)
    print("Style weight:", args.sw)
    print("Content weight:", args.cw)
    print("\u001b[0m")

    # Call run() and display the output image
    output_image = run(cnn, mean, std, content, \
            style, input_image, content_layers, \
            style_layers, args.sw, args.cw, epochs=300)

    plt.figure()
    show_img(output_image, name='Output image')
    plt.ioff()
    plt.show()

    # Save the output image in the current directory
    print("\u001b[1m\033[92mSaving output image to {}\033[0m".format(args.output))
    final_img = output_image[0]
    save_image(final_img, args.output)


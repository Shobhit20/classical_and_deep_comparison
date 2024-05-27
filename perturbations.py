import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage.util import random_noise
import random


torch.manual_seed(42)

def scale_image(image):
    # Bring values in the range of 0 to 255
    scaled_image = torch.clamp(image, 0, 255)
    
    # Convert to integer tensor
    scaled_image = scaled_image.round()/255
    return scaled_image

def add_gaussian_noise(image_tensor, std_dev):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Convert the tensor to numpy array
    image_np = image_tensor.numpy()
    # Generate Gaussian noise with the given standard deviation
    noise = np.random.normal(loc=0, scale=std_dev, size=image_np.shape)
    # Add the noise to the image
    noisy_image = image_np + noise
    noisy_image = torch.from_numpy(noisy_image).type(torch.float32)
    # Ensure pixel values are integers in the range 0..255
    return scale_image(noisy_image)


def gaussian_blur(image_tensor, num_iterations):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Assuming your input image is of shape [batch_size, channels, height, width]
    input_image = image_tensor.unsqueeze(0)  # Example shape [batch_size, 3 channels, 32x32]
    # Assuming your kernel is of shape [out_channels, in_channels, kernel_height, kernel_width]
    kernel_channel = torch.tensor([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]], dtype=torch.float32) / 16

    # Apply the convolution operation separately to each channel
    for iteration in range(num_iterations):
        conv_outputs = []
        for i in range(input_image.shape[1]):  # Iterate over each input channel
            input_channel = input_image[:, i:i+1, :, :].to(torch.float32)  # Select the current input channel
            conv_output = F.conv2d(input_channel, kernel_channel.unsqueeze(0).unsqueeze(0), padding='same')  # Apply convolution
            conv_outputs.append(conv_output)
        # Concatenate the outputs along the channel dimension
        input_image = torch.cat(conv_outputs, dim=1).to(torch.float32)
    
    # Now output_image will have shape [batch_size, channels, height, width]
    return scale_image(input_image.squeeze(0))

def contrast_change(image_tensor, factor):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Contrast simply means multiplication by factor
    contrast_image = image_tensor * factor
    return scale_image(contrast_image)

def brightness_change(image_tensor, factor):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Brightness means adding or subtracting factor from all pixels
    brightness_image = image_tensor + factor
    return scale_image(brightness_image)

def place_square_occlusion(image_tensor, size):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Randomly select the top-left corner of the square region
    x = random.randint(0, image_tensor.shape[1] - size - 1)
    y = random.randint(0, image_tensor.shape[2] - size - 1)
    # Set the pixels in the square region to zero
    image_tensor[:, x:x+size, y:y+size] = 0
    return scale_image(image_tensor)

def add_salt_and_pepper_noise(image_tensor, noise_strength):
    # Converting image to range (0,255)
    image_tensor = (image_tensor * 255)
    # Convert image to numpy array and normalize to [0, 1]
    image_np = image_tensor.float().permute(1, 2, 0).numpy() / 255.0
    # Add salt and pepper noise
    noisy_image_np = random_noise(image_np, mode='s&p', amount=noise_strength)
    # Convert back to torch tensor and rescale to [0, 255]
    noisy_image = torch.tensor(noisy_image_np * 255, dtype=torch.uint8).permute(2, 0, 1)
    return scale_image(noisy_image)

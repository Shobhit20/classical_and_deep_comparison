import torchvision.transforms as transforms
from perturbations import *

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(512, scale=(0.8, 1)),
    transforms.ToTensor(),
])

# Define transform for test/validation (without augmentation)
val_transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])

# Model contants
TRAIN_DIR = 'dataset/train'
TEST_DIR = 'dataset/test'
TRAIN_VAL_SPLIT = 0.9
CLASSICAL_MODEL_DIR = "models/classical/"
DEEP_MODEL_DIR = "models/deepmodel/"
CLUSTERING_MODEL_FILE = "clustering_model_scale.pkl"
SVC_MODEL_FILE = "svc_model_scale.pkl"
PCA_MODEL_FILE = "pca_model_scale.pkl"
DEEP_MODEL_FILE = "resnet_18_scale.pth"

# perturbation values
gaussian_noise = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18 ]
gaussian_blurring = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
contrast_inc = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25 ]
contrast_dec = [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
brightness_inc = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
brightness_dec = [-x for x in brightness_inc]
occlusion_size = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
sap_noise = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]


# perturbation dictionary with values and functions
perturbations_dict = {"Gaussian Noise": (gaussian_noise, add_gaussian_noise), \
                "Gaussian Blurring": (gaussian_blurring, gaussian_blur), \
                "Contrast Increase": (contrast_inc, contrast_change), \
                "Contrast Decrease": (contrast_dec, contrast_change), \
                "Brightness Increase": (brightness_inc, brightness_change), \
                "Brightness Decrease": (brightness_dec, brightness_change), \
                "Occlusion Of Image": (occlusion_size, place_square_occlusion), \
                "Salt and Pepper Noise": (sap_noise, add_salt_and_pepper_noise)}
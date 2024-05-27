from skimage import color, io, feature
from skimage import img_as_float
import cv2
from torchvision.datasets import ImageFolder
from constants import *
from torch.utils.data import DataLoader, random_split
import torch
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle
import tqdm
import os

torch.manual_seed(42)
sift = cv2.SIFT_create()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def create_directory_if_not_exists(directory):
    """
    Check if a directory exists, if not, create it.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")


def load_datasets(train_dir, test_dir, train_val_split, batch_size=32):
    # Load the training dataset
    dataset = ImageFolder(root=train_dir)
    class_names = dataset.classes
    # get train size
    train_size = int(train_val_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Apply transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    # Make data loader for train and val
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # Load the testing dataset
    test_dataset = ImageFolder(test_dir, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, class_names

def extract_features(data_loader):
    features = []
    labels = []
    for batch in data_loader:
        batch_data, batch_labels = batch
        # Extract features from each image in the batch separately
        for image in batch_data:
            np_image = image.permute(1, 2, 0).numpy()
            pil_image = Image.fromarray((np_image * 255).astype(np.uint8))
            features.append(pil_image)
        labels.extend(batch_labels.tolist())
    return features, torch.tensor(labels).numpy()

#Extract HoG features from an image(512X512)
def extract_hog_features(image, pixels_per_cell=(32, 32), cells_per_block=(2, 2)):
    image = img_as_float(image)
    if image.ndim == 3: 
        image = color.rgb2gray(image)  #grayscale conversion
    hog_features = feature.hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
    return hog_features

def extract_sift_features(image):
    numpy_image = np.array(image)
    # Convert the color space to grayscale using OpenCV
    grayscale_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
    # Extract sift features
    _, des = sift.detectAndCompute(grayscale_image, None)
    if des is not None:
        return des
    return None

def load_sift_features(images):
    sift_features = []
    for image in images:
        descriptor = extract_sift_features(image)
        if descriptor is not None:
            sift_features.extend(descriptor)
    
    return np.array(sift_features)

def make_feature_train_ready(kmeans_model, images):
    sift_histograms, hog_features = [], []
    for i, image in tqdm.tqdm(enumerate(images)):
        des = extract_sift_features(image)
        if des is not None:
            # perform kmeans on sift features
            visual_words = kmeans_model.predict(des)
            histogram, _ = np.histogram(visual_words, bins=range(101))
            sift_histograms.append(histogram)
        else:
            sift_histograms.append(np.zeros(100))
        # get all hog features
        hog_feature = extract_hog_features(image)
        hog_features.append(hog_feature)
    return sift_histograms, hog_features

# Save model files
def save_sklearn_model(model, key):
    with open(key, "wb") as f:
        pickle.dump(model, f)

# Load model files
def load_sklearn_model(key):
    with open(key, "rb") as f:
        model = pickle.load(f)
    return model
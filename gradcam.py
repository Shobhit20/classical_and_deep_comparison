from resnet_model import model_backbone
from utils import load_datasets
import json
import torch
from torchvision import models, transforms
from PIL import Image as PilImage
from perturbations import *
from omnixai.data.image import Image
from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM
from constants import *

# Load the pre-trained ResNet model
train_loader, val_loader, test_loader, class_names = load_datasets(TRAIN_DIR, TEST_DIR, TRAIN_VAL_SPLIT)

model = model_backbone(DEEP_MODEL_DIR + DEEP_MODEL_FILE, class_names, load_model=True)
model.eval()  # Set the model to evaluation mode

print("Model loaded")
image_path = 'tennis.jpg'  # Replace 'your_image.jpg' with the path to your image
img = Image(PilImage.open(image_path).convert('RGB'), )
preprocess = lambda ims: torch.stack([transform(im.to_pil()) for im in ims])


transform =  transforms.Compose([
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: place_square_occlusion(x,100)), # replace with perturbation you want
            ])# Load and preprocess the image
explainer = GradCAM(
    model=model,
    target_layer=model.layer4[0].conv2,
    preprocess_function=preprocess
)

# Explain the top label
explanations = explainer.explain(img)
print("Predicted class", class_names[explanations.get_explanations()[0]['target_label']])
explanations.ipython_plot(index=0, class_names=class_names)

import torch
from utils import load_datasets, create_directory_if_not_exists
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, f1_score
from plotting import plot_confusion_matrix, plot_training_metrics
from constants import *


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def model_backbone(model_path, class_names, load_model=False):
    """
    Loads the model architecture and weights if needed

    Parameters:
        model_path (str): path at which the model is kept if needs to be loaded
        class_names (list):  The list of class names i.e. categories of classification
        load_model (bool): True if you want to load existing weights
    
    Return:
        model: torch model object for resnet
    """
    model = resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))  # Modify the output layer to match the number of classes
    model = model.to(device)
    if load_model:
        model.load_state_dict(torch.load(model_path))
    return model

def evaluate_on_test(model_path, test_loader, class_names, load_model):
    """
    The function helps run evaluations on test files

    Parameters:
        model_path (str): path at which the model is kept if needs to be loaded
        test_loader (Torch DataLoader): dataloader which contains the data on which the model needs to be evaluated
        class_names (list):  The list of class names i.e. categories of classification
        load_model (bool): True if you want to load existing weights
    
    Return:
        model: torch model object for resnet
    """
    model = model_backbone(model_path, class_names, load_model=load_model)
    model.eval()  # Set the model to evaluation mode
    test_labels, predictions = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            # Running the images through model
            outputs = model(images)
            # Extracting the top prediction
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())  # Convert predictions to CPU and append
            test_labels.extend(labels.cpu().numpy())  # Convert labels to CPU and append

    # Generate classification report
    test_labels_readable = [class_names[d] for d in test_labels]
    predictions_readable = [class_names[d] for d in predictions]

    # Calculate accuracy
    accuracy = accuracy_score(test_labels_readable, predictions_readable)
    f1 = f1_score(test_labels_readable, predictions_readable, average='weighted')
    report = classification_report(test_labels_readable, predictions_readable)
    plot_confusion_matrix(test_labels_readable, predictions_readable, class_names, "comf_mat_deep.png")
    print("Accuracy of the ResNet model:", accuracy)
    print("F1 score of the ResNet model:", f1)
    print("Model classifcation report")
    print(report)
    return model

def train_model(train_loader, val_loader, test_loader, model_path, class_names, num_epochs=10):
    """
    The function helps train the resnet model

    Parameters:
        train_loader (Torch DataLoader): dataloader which contains the data on which the model needs to be trained
        val_loader (Torch DataLoader): dataloader which contains the data on which the model needs to be validated
        test_loader (Torch DataLoader): dataloader which contains the data on which the model needs to be evaluated
        model_path (str): path at which the model is kept if needs to be loaded
        class_names (list):  The list of class names i.e. categories of classification
        num_epochs (int): The number of epochs to train
    
    Return:
        model: torch model object for resnet
    """
    model = model_backbone(model_path, class_names)
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_list, acc_list = [],[]
    best_val_loss, best_model_state = float('inf'), None
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Calculate validation accuracy
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        loss_list.append([train_loss, val_loss])
        acc_list.append([train_accuracy, val_accuracy])
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Check if current validation loss is better than the best seen so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            # Save the best model checkpoint
            print("Saving the new best model")
            torch.save(best_model_state, model_path)

    print("Saving training graph!!")
    plot_training_metrics(loss_list, acc_list, num_epochs, "training_graph.png")
    model = evaluate_on_test(model_path, test_loader, class_names,load_model=True)
    return model
    
def load_or_train_resnet_model(model_dir, model_file, load_checkpoint=True):
    """
    The driver function to facilitate resnet modules

    Parameters:
        model_dir (str): the directory at which the model is kept
        model_file (str): the name of the model file either to be loaded or to be saved by
        load_checkpoint (bool): the variable which defines whether to laod or retrain model

    Returns:
        resnet_model: torch model object for resnet
    """
    create_directory_if_not_exists(model_dir)
    train_loader, val_loader, test_loader, class_names = load_datasets(TRAIN_DIR, TEST_DIR, TRAIN_VAL_SPLIT)

    model_path = model_dir + model_file
    if load_checkpoint:
        resnet_model = model_backbone(model_path, class_names, load_checkpoint)
    else:
        resnet_model = train_model(train_loader, val_loader, test_loader, \
                                model_path, class_names)
    return resnet_model


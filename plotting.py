import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import math
from constants import *

def plot_confusion_matrix(true_labels_numeric, predicted_labels_numeric, classes, filename):
    conf_matrix = confusion_matrix(true_labels_numeric, predicted_labels_numeric)

    # Plot confusion matrix
    sns.set(font_scale=1.2)  # Adjust font scale as needed
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False, square=True,
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def plot_training_metrics(loss, acc, epochs_num,filename):
    epochs = list(range(10))
    loss_train = [sublist[0] for sublist in loss]
    loss_val = [sublist[1] for sublist in loss]
    accuracy_train = [sublist[0] for sublist in acc]
    accuracy_val = [sublist[1] for sublist in acc]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss in the first subplot
    ax1.plot(epochs, loss_train, marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=8, label='Training Loss')    
    ax1.plot(epochs, loss_val, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8, label="Validation Loss")
    
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(False)
    ax1.legend()

    # Plot accuracy in the second subplot
    ax2.plot(epochs, accuracy_train, marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=8, label="Train Accuracy")    
    ax2.plot(epochs, accuracy_val, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8, label='Validation Accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(False)
    ax2.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot
    plt.savefig(filename)

    # Don't forget to close the plot to release resources
    plt.close()


def plot_perturbation_graphs(accuracy_dict_classical, accuracy_dict_deep, filename):

    perturbation_names = list(perturbations_dict.keys())
    # Determine the number of models
    num_models = len(perturbation_names)
    # Calculate the number of rows and columns for the grid layout
    num_rows = math.ceil(num_models / 2)
    num_cols = 2
    fig_labels_name = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)']
    # Create subplots with better color choices
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 6*num_rows))
    # Create subplots

    # Iterate over the dictionary entries and plot the accuracies on each subplot

    for i, perturbation_name in enumerate(perturbation_names):
        row = i // num_cols
        col = i % num_cols
        x_values = perturbations_dict[perturbation_name][0]
        accuracies_classical = accuracy_dict_classical[perturbation_name]
        accuracies_deep = accuracy_dict_deep[perturbation_name]
        epoch_labels = [str(x) for x in x_values]
        axs[row, col].plot(epoch_labels, accuracies_classical, marker='o', linestyle='-', color='mediumseagreen', linewidth=2, markersize=8)
        axs[row, col].plot(epoch_labels, accuracies_deep, marker='o', linestyle='-', color='teal', linewidth=2, markersize=8)
        axs[row, col].set_title(fig_labels_name[i] + " "+ perturbation_name, fontsize=14, fontweight='bold', color='black', y=-0.3)
        axs[row, col].set_xlabel('Perturbation factor', fontsize=12, fontweight='bold', color='darkgreen')
        axs[row, col].set_ylabel('Accuracy', fontsize=12, fontweight='bold', color='darkgreen')
        axs[row, col].tick_params(axis='both', which='major', labelsize=10)
        axs[row, col].spines['top'].set_visible(False)
        axs[row, col].spines['right'].set_visible(False)
        # axs[row, col].legend()
        axs[row, col].legend(['Classical Model', 'Deep Model'], loc='upper right', bbox_to_anchor=(1.0, 1.05), borderaxespad=0)

    plt.subplots_adjust(hspace=0.5)
    # Add a title for the entire figure
    plt.grid(False)
    # Adjust layout
    plt.savefig(filename)

    # Don't forget to close the plot to release resources
    plt.close()
from svc_model import load_or_train_svc_model
from resnet_model import load_or_train_resnet_model
from utils import load_datasets, extract_features, make_feature_train_ready
from perturbations import *
from torchvision import transforms
from tqdm import tqdm
from constants import *
from sklearn.metrics import accuracy_score
from plotting import plot_perturbation_graphs
import argparse


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class TestLoaderEvaluator:
    def __init__(self, test_loader, values, transformation_function, perturbation_name):
        self.test_loader = test_loader
        self.values = values
        self.transformation_function = transformation_function
        self.perturbation_name = perturbation_name

    def evaluate_deepmodel(self, model):
        """
        The function runs the deep learning model on the test data loader containing some perturbation

        Parameters:
            model (model object) - the ResNet model itself

        Returns
            accuracy (float) - the accuracy of the model on the testloader
        """
        test_labels, predictions = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(device), labels.to(device)

                # Run the images through our model
                outputs = model(images)

                # Selecting the best output from the predictions
                _, predicted = outputs.max(1)
                predictions.extend(predicted.cpu().numpy())  # Convert predictions to CPU and append
                test_labels.extend(labels.cpu().numpy())  # Convert labels to CPU and append
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, predictions)*100
        return accuracy

    def evaluate_classicalmodel(self, clustering_model, pca_model, svc_model ):
        """
        The function runs the classical learning model on the test data loader containing some perturbation

        Parameters:
            clustering_model (pickle object) - the kmeans clustering model for sift features
            pca_model (pickle object) - the pca dimension reducing model for hog features
            svc_model (pickle object) - the SVM model which does the classification of the inage

        Returns
            accuracy (float) - the accuracy of the model on the testloader
        """
        # Extract images and labels
        test_images, test_labels = extract_features(self.test_loader)
        # Extract sift and hog features
        sift_test, hog_test = make_feature_train_ready(clustering_model, test_images)
        # Compress hog features
        test_hog_compressed = pca_model.transform(hog_test)
        # Combine the hog and sift features
        test_features = np.concatenate((test_hog_compressed, sift_test), axis=1)
        # Running predictions
        predictions = svc_model.predict(test_features)
        accuracy = accuracy_score(test_labels, predictions)*100
        return accuracy

    def evaluate(self, resnet_model,  clustering_model, pca_model, svc_model ):
        """
        Driver evaluation function which makes calls to both evaluation functions

        Parameters:
            resnet_model (model object) - the ResNet model itself
            clustering_model (pickle object) - the kmeans clustering model for sift features
            pca_model (pickle object) - the pca dimension reducing model for hog features
            svc_model (pickle object) - the SVM model which does the classification of the inage
        
        Returns
            accuracy_classical (list) - list of all accuracy for classical model for all perturbations
            accuracy_deep (list) - list of all accuracy for deep model for all perturbations

        """
        accuracy_classical, accuracy_deep = [], []
        for value in self.values:
            correct = 0
            total = 0
            self.test_loader.dataset.transform =  transforms.Compose([
                # crop the image to make it 512 * 512
                transforms.Resize(512),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: self.transformation_function(x, value)), # perturbation function
            ])

            # Evaluate the classical model
            test_accuracy_classical = self.evaluate_classicalmodel(clustering_model, pca_model, svc_model )
            accuracy_classical.append(test_accuracy_classical)
            print(f'Test Accuracy with Classical model on {self.perturbation_name} factor {value}: {test_accuracy_classical:.2f}%')

            # Evaluate the deep learning model
            test_accuracy_deep = self.evaluate_deepmodel(resnet_model)
            accuracy_deep.append(test_accuracy_deep)
            print(f'Test Accuracy with deep model on {self.perturbation_name} factor {value}: {test_accuracy_deep:.2f}%')
        return accuracy_classical, accuracy_deep


def main(args):
    train_loader, val_loader, test_loader, class_names = load_datasets(TRAIN_DIR, TEST_DIR, TRAIN_VAL_SPLIT)
    # Load or train a new classical model
    clustering_model, pca_model, svc_model = load_or_train_svc_model(CLASSICAL_MODEL_DIR, CLUSTERING_MODEL_FILE, \
                                            PCA_MODEL_FILE, SVC_MODEL_FILE, load_checkpoint=args.load_from_checkpoint_classical, \
                                            grid_search=args.grid_search)

    # Load or train the deep learning model
    resnet_model = load_or_train_resnet_model(DEEP_MODEL_DIR, DEEP_MODEL_FILE, load_checkpoint=args.load_from_checkpoint_deep)
    print("Models Loaded!!")
    if args.perturbation_study:
        accuracy_dict_classical, accuracy_dict_deep = {}, {}
        for key, value in perturbations_dict.items():

            perturbation, modification_function = value[0], value[1]

            # Evaluate the model with different perturbation and perturbation factors
            evaluator = TestLoaderEvaluator(test_loader, perturbation, modification_function, key)
            accuracy_dict_classical[key], accuracy_dict_deep[key] = evaluator.evaluate(resnet_model, \
                                            clustering_model, pca_model, svc_model )
        # Plot the output graph of accuracy with perturbation factor
        plot_perturbation_graphs(accuracy_dict_classical, accuracy_dict_deep, "perturbation_plots.png")


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="A script to parse command-line arguments")

    # Add argument for script name
    parser.add_argument('--load_from_checkpoint_classical', help='False for retraining', default=True)
    parser.add_argument('--grid_search', help='Do grid search if retraining', default=True)
    parser.add_argument('--load_from_checkpoint_deep', help='False for retraining', default=True)
    parser.add_argument('--perturbation_study', help='True if we want to retrain classical', default=True)

    # Add argument for additional arguments with identifier
    parser.add_argument('--arguments', nargs='*', help='Additional arguments', dest='arguments')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with parsed arguments
    main(args)
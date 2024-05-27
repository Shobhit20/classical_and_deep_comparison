from utils import extract_features, load_datasets, \
    extract_hog_features, extract_sift_features, load_sift_features, \
    make_feature_train_ready, save_sklearn_model, load_sklearn_model, \
    create_directory_if_not_exists
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plotting import plot_confusion_matrix
from constants import * 

TRAIN_VAL_SPLIT = 0.9999


def svc_trainer(train_features, test_features, train_labels, test_labels, class_names, grid_search):
    """
    The function which does the training for our svc model

    Parameters:
        train_features (array): Training data with features extracted 
        test_features (array): Test data with features extracted
        train_labels (array): Correct labels for the training images
        test_labels (array): Correct labels for the test images
        class_names (list): The list of class names i.e. categories of classification

    Return:
        clf (sklearn model object): the classifier model for classification
        accuracy (float): accuracy of the classification model
        f1 (float): f1 score of our classification model
        report (str): the report of the entire classification for each class
    """

    # Grid PARAM SEARCH
    if grid_search:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC())
        ])

        # Define parameter grid for grid search
        param_grid = {
            'svc__C': [0.1, 1, 10],  # Regularization parameter
            'svc__kernel': ['rbf', 'poly'],  # Kernel type
            'svc__gamma': ['scale', 'auto'],  # Kernel coefficient
        }

        # Perform Grid Search with 5-fold cross-validation
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid,verbose=2, cv=5)
        grid_search.fit(train_features, train_labels)

        # Best parameters found
        best_params = grid_search.best_params_
        print("Best parameters:", best_params)
        clf = grid_search.best_estimator_
    else:
        # Train model with default settings without param search
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='scale', C=10))
        clf.fit(train_features, train_labels)

    predictions = clf.predict(test_features)

    # Convert labels from indexes to actual class names
    test_labels_readable = [class_names[d] for d in test_labels]
    predictions_readable = [class_names[d] for d in predictions]

    # Calculate Metrics
    accuracy = accuracy_score(test_labels_readable, predictions_readable)
    f1 = f1_score(test_labels_readable, predictions_readable, average='weighted')
    report = classification_report(test_labels_readable, predictions_readable)
    plot_confusion_matrix(test_labels_readable, predictions_readable, class_names, "comf_mat_classical.png")

    return clf, accuracy, f1, report



def train_model(clusering_model_dir, pca_model_dir, svc_model_dir, grid_search):
    """
    The model trains the clustering model for sift and the dimensionality reduction model for hog

    Parameters:
        clusering_model_dir (string): The directory where we want to save kmeasns model
        pca_model_dir (string): The directory where we want to save pca model
        svc_model_dir (string): The directory where we want to save svc model
        grid_search (bool): The parameter decides if we want to use default model for svc or do grisearch

    Return:
        kmeans (sklearn model object): the kmeans model for sift
        pca (sklearn model object): the pca model for hog
        best_svc_model (sklearn model object): the best classifier model for classification
    """
    train_loader, val_loader, test_loader, class_names = load_datasets(TRAIN_DIR, TEST_DIR, TRAIN_VAL_SPLIT)
    print("Loading train and test dataset")
    train_images, train_labels = extract_features(train_loader)
    print("Training data loaded!!")
    test_images, test_labels = extract_features(test_loader)
    print("Test data loaded!!")

    print("Extracting SIFT features for kmeans, This may take 5-10 mins time")
    sift_features = load_sift_features(train_images)
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(sift_features)
    save_sklearn_model(kmeans, clusering_model_dir)
    print("CLustering Model Saved!!")

    print("Extracting HOG and SIFT features")
    sift_train, hog_train = make_feature_train_ready(kmeans, train_images)
    sift_test, hog_test = make_feature_train_ready(kmeans, test_images)

    print("Compressing HOG features using PCA")
    pca = PCA(n_components=100)  # Set the desired number of components
    pca.fit(hog_train)
    train_hog_compressed = pca.transform(hog_train)
    test_hog_compressed = pca.transform(hog_test)
    save_sklearn_model(pca, pca_model_dir)
    print("PCA Model Saved!!")

    train_combined = np.concatenate((train_hog_compressed, sift_train), axis=1)
    test_combined = np.concatenate((test_hog_compressed, sift_test), axis=1)

    print("Training the SVC model")
    best_svc_model, accuracy_final, f1_final, report_final = svc_trainer(train_combined, \
                        test_combined, train_labels, test_labels, class_names, grid_search)
    print("Accuracy of the SVC model:", accuracy_final)
    print("F1 score of the SVC model:", f1_final)
    print("Model classifcation report")
    print(report_final)
    save_sklearn_model(best_svc_model, svc_model_dir)
    print("Best SVC model saved!!")
    return kmeans, pca, best_svc_model



def load_or_train_svc_model(model_dir, clustering_model_dir, pca_model_dir, \
        svc_model_dir, load_checkpoint=True, grid_search=False):
    """
    The model is driver function to facilitate the entire svc classification

    Parameters:
        model_dir (str): Where the model files are kept 
        clustering_model_dir (str): Filename to load or save by for clustering
        pca_model_dir (str): Filename to load or save by for pca
        svc_model_dir (str): Filename to load or save by for classification
        load_checkpoint (bool): Whether to load model or retrain
        grid_search (bool): Whether to do grid search or use default model

    Return:
        clustering_model (sklearn model object): the kmeans model for sift
        pca_model (sklearn model object): the pca model for hog
        svc_model (sklearn model object): the best classifier model for classification
    """
    create_directory_if_not_exists(model_dir)

    # Extracting directories
    clustering_model_dir = model_dir + clustering_model_dir
    pca_model_dir = model_dir + pca_model_dir
    svc_model_dir = model_dir + svc_model_dir

    # check for loading a model or re-training
    if load_checkpoint:
        clustering_model = load_sklearn_model(clustering_model_dir)
        pca_model = load_sklearn_model(pca_model_dir)
        svc_model = load_sklearn_model(svc_model_dir)
    else:
        clustering_model, pca_model, svc_model = train_model(clustering_model_dir, \
                                                    pca_model_dir, svc_model_dir, grid_search)
    return clustering_model, pca_model, svc_model 



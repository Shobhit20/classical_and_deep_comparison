# classical_and_deep_comparison

The main file which does the training of both models and runs the model through perturbations.

## Instructions to run
python main.py --load_from_checkpoint_classical True --grid_search True --load_from_checkpoint_deep True --perturbation_study True

load_from_checkpoint_classical - If this argument is set to True, we load the model from checkpoints defined in
constants.py file else it retrains the model
grid_search - If this argument is set to True, we perform a grid search else we train the model with rbf kernel and C=10
load_from_checkpoint_deep - If this argument is set to True, we load the model from checkpoints defined in constants.py file else it retrains the model
perturbation_study - If the argument is True, we perform a perturbation study for our model, else we do not do any study for perturbations

## File descriptions

1. Driver file - main.py
The main file which does the training of both models and runs the model through perturbations.

2. Classical model file - svc_model.py
This file trains the SVC model if load_from_checkpoint_classical is set to False. Also does a grid search if the grid_search parameter is set to True

3. Deep model file - resnet_model.py
This file trains the ResNet model if load_from_checkpoint_deep is set to False.

4. Perturbations file - perturbations.py
This file contains all the perturbation functions used to induce perturbations on the image.

6. Utility file - utils.py
This file is the primary utility file being used to generate datasets, load models, save models, extracting features, etc.

7. Plotting file - plotting.py
This file helps plot the perturbation plot of accuracy with respect to the factor of perturbation. Also helps in plotting the training metric graphs and the confusion matrices.

8. Grad-CAM visualization file - gradcam.py
This file helps in generating a Grad-CAM image.

## Report and analysis outcome 
The report for the analysis can be found [here](https://github.com/Shobhit20/classical_and_deep_comparison/blob/main/IVC%20report.pdf)



# classical_and_deep_comparison

The main file which does the training of both models and runs the model through perturbations.

## Instructions to run
python main.py --load_from_checkpoint_classical True --grid_search True --load_from_checkpoint_deep True
,→ --perturbation_study True

load_from_checkpoint_classical - If this argument is set to True, we load the model from checkpoints defined in
constants.py file else it retrains the model
grid_search - If this argument is set to True, we perform a grid search else we train the model with rbf kernel and C=10
load_from_checkpoint_deep - If this argument is set to True, we load the model from checkpoints defined in constants.py file else it retrains the model
perturbation_study - If the argument is True, we perform a perturbation study for our model, else we do not do any study for perturbations

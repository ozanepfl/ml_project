import argparse

import numpy as np
import torch
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, Trainer
from src.utils import normalize_fn, accuracy_fn, get_n_classes
np.random.seed(100)


#============================================================================================================================
# Below is the code to find the optimal parameters for MLP
# Hidden unit numbers per layer, activation functions, learning rates, max iterations, and batch sizes are observed
# The results are written to a text file
#============================================================================================================================

def log_results(filename, params, acc):
    """ Log experiment results to a text file.
    
    Args:
        filename (str): The path to the log file.
        params (dict): A dictionary containing the parameters used for the experiment.
        acc (int): Accuracy of testing
    """
    with open(filename, 'a') as f:
        f.write(f"Parameters: {params}\n")
        f.write(f"Accuracy: {acc}\n")
        f.write("--------------------------------------------------\n")

hidden_unit_list = [[128], [128, 128], [128, 128, 128], [128, 128, 128, 128],
                    [256], [256, 256], [256, 256, 256], [512], [512, 512], [512, 512, 512],
                    [512, 256, 256], [512, 256, 128], [1024, 512, 256], [1024, 512, 256, 128]]

activation_list1 = [[0], [1], [2]]
activation_list2 = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]
activation_list3 = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2],
                    [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2],
                    [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]

lr_list = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]

maxiter_list = [5, 20, 50, 100, 150]

batch_list = [10, 30, 64, 100, 600]

def optimal_hidden_units(xtrain, xtest, ytrain, y_test):
    for hu in hidden_unit_list:
        parameter_comparison_mlp(hu, [0], 1e-1, 5, 64, xtrain, xtest, ytrain, y_test)

def optimal_activation(xtrain, xtest, ytrain, y_test):
    for ac in activation_list1:
        parameter_comparison_mlp([128], ac, 1e-1, 5, 64, xtrain, xtest, ytrain, y_test)
    for ac in activation_list2:
        parameter_comparison_mlp([128, 128], ac, 1e-1, 5, 64, xtrain, xtest, ytrain, y_test)
    for ac in activation_list3:
        parameter_comparison_mlp([128, 128, 128], ac, 1e-1, 5, 64, xtrain, xtest, ytrain, y_test)                
    
def optimal_learning_rate(xtrain, xtest, ytrain, y_test):
    for lr in lr_list:
        parameter_comparison_mlp([128, 128], [0], lr, 5, 64, xtrain, xtest, ytrain, y_test)

def optimal_maxiter(xtrain, xtest, ytrain, y_test):
    for mi in maxiter_list:
        parameter_comparison_mlp([128, 128], [0], 1e-1, mi, 64, xtrain, xtest, ytrain, y_test)

def optimal_batch_size(xtrain, xtest, ytrain, y_test):
    for b in batch_list:
        parameter_comparison_mlp([128, 128], [0], 1e-1, 5, b, xtrain, xtest, ytrain, y_test)


def parameter_comparison_mlp(hu, ac, lr, mi, b, xtrain, xtest, ytrain, y_test):
    n_classes = get_n_classes(ytrain)

    model = MLP(xtrain.shape[1], n_classes,
                    hidden_units=hu,
                    activations=ac)

    method_obj = Trainer(model, lr=lr, epochs=mi, batch_size=b)

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)
    acc = accuracy_fn(preds, y_test)

    params = {
        "hu": hu,
        "ac": ac,
        "lr": lr,
        "mi": mi,
        "b": b
    }

    # Log the testing results
    log_results("training_log.txt", params, acc)
    return

def main():
    xtrain, xtest, ytrain = load_data("C:\\Users\\Asus\\ml_project\\sciper1_sciper2_sciper3_project 2\\dataset")
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    # Shuffle the data before processing 
    indices = np.arange(xtrain.shape[0])
    np.random.permutation(indices)
    xtrain = xtrain[indices, :]
    ytrain = ytrain[indices]

    # Validation
    ratio = 0.8 
    num_samples = xtrain.shape[0]
    n_train = int(num_samples * ratio)
    x_train = xtrain[:n_train] 
    xtest = xtrain[n_train:]
    y_train = ytrain[:n_train]
    y_test = ytrain[n_train:]
    xtrain = x_train
    ytrain = y_train

    # Normalization the data before training and prediction 
    mean_val = np.mean(xtrain, keepdims=True)
    std_val = np.std(xtrain, keepdims=True)
    xtrain = normalize_fn(xtrain, mean_val, std_val)
    xtest = normalize_fn(xtest, mean_val, std_val)

    optimal_hidden_units(xtrain, xtest, ytrain, y_test)
    optimal_activation(xtrain, xtest, ytrain, y_test)            
    optimal_learning_rate(xtrain, xtest, ytrain, y_test)
    optimal_maxiter(xtrain, xtest, ytrain, y_test)
    optimal_batch_size(xtrain, xtest, ytrain, y_test)
 
#===========================================================================================================================
#===========================================================================================================================

if __name__ == '__main__':
    main()
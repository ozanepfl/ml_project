import argparse

import numpy as np
import torch
from torchinfo import summary

from src.data import load_data
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer, MyViT
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)

    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    # Shuffle the data before processing 
    indices = np.arange(xtrain.shape[0])
    np.random.permutation(indices)
    xtrain = xtrain[indices, :]
    ytrain = ytrain[indices]

    # Make a validation set
    if not args.test:
        # We are separating the dataset into a validation set (20%) and a training set (80%)
        # No test set values are used during the training process
        ratio = 0.8 
        num_samples = xtrain.shape[0]
        n_train = int(num_samples * ratio)
        x_train = xtrain[:n_train] 
        xtest = xtrain[n_train:]
        y_train = ytrain[:n_train]
        y_test = ytrain[n_train:]
        xtrain = x_train
        ytrain = y_train

        
    # No need to add bias, because the deep network does itself 
    
    # Normalization the data before training and prediction 

    mean_val = np.mean(xtrain, keepdims=True)
    std_val = np.std(xtrain, keepdims=True)
    xtrain = normalize_fn(xtrain, mean_val, std_val)
    xtest = normalize_fn(xtest, mean_val, std_val)


    ### WRITE YOUR CODE HERE
    print("Using PCA")

    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        pca_obj.find_principal_components(xtrain)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(xtrain.shape[1], n_classes,
                    hidden_units=args.hidden_units,
                    activations=args.activations)

    if args.nn_type == "cnn": 
        input_channel = 1 #grey scaled
        xtrain = xtrain.reshape(xtrain.shape[0], 1, 28, 28)
        xtest = xtest.reshape(xtest.shape[0], 1, 28, 28)
        model = CNN(input_channel, n_classes)
        #print("model shape is ", model(torch.randn(args.nn_batch_size, 1, 28, 28)).shape)

    if args.nn_type == "transformer": 
        model = MyViT(...)
        
    summary(model, input_size=(args.nn_batch_size, 1, 28, 28))

    # Trainer object
    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, y_test)
    macrof1 = macrof1_fn(preds, y_test)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps'")
    parser.add_argument('--use_pca', action="store_true", help="use PCA for feature reduction")
    parser.add_argument('--pca_d', type=int, default=100, help="the number of principal components")


    parser.add_argument('--lr', type=float, default=1e-10, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=5, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")

    # extra arguments for MLP
    parser.add_argument('--hidden_units', type=int, default=[128], nargs='+', help="size of each layer (input and output layer are excluded)")
    parser.add_argument('--activations', type=int, default=[0], nargs='+', help="activation function/s to be used in each layer")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
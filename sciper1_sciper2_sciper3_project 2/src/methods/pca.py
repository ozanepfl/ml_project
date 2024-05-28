import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        # Center the data with the mean
        training_centered = training_data - self.mean
        # Create the covariance matrix
        covariance = np.cov(training_centered, rowvar=False)
        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigen_values, eigen_vectors = np.linalg.eigh(covariance)
        # Choose the top d eigenvalues and corresponding eigenvectors. 
        # Hint: sort the eigenvalues (with corresponding eigenvectors) in decreasing order first.
        decreasing_indices = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[decreasing_indices]
        eigen_vectors = eigen_vectors[:, decreasing_indices]

        self.W = eigen_vectors[:, :self.d]
        d_biggest_egval = eigen_values[:self.d]

        # Compute the explained variance
        exvar = np.sum(d_biggest_egval) / np.sum(eigen_values) * 100
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        # project the data using W
        data_centered = data - self.mean
        data_reduced = data_centered @ self.W
        return data_reduced
        


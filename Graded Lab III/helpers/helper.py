import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 

def plotC(trials, D, list_of_N, fractions):
    plt.figure()
    domain = np.round(list_of_N)/D
    plt.plot(domain,fractions)
    plt.xlabel("N/D")
    plt.ylabel("C(N,D)")
    plt.title("Fraction of convergences per {} trials as a function of N".format(trials))
    
def sample_data(N, D, seed):
    """
    Generate the synthetic data.
    
    parameters:
        N: (int) number of samples
        D: (int) number of dimensions
    
    output:
        X (np.array): is of shape (N, D), filled with numbers drawn from bernoulli distribution
        y (np.array): is of size (N), filled with numbers drawn from bernoulli distribution
    """
    
    k, lamb = 1, 0.5
    
    X = np.empty((N, D)) # X is [N x D]
    y = np.empty(N)      # y is [N]
   
    # generate one observation by drawing 10 samples 
    # from a Bernoulli distribution (Binomial with k=1)
    np.random.seed(seed)
    X = np.random.binomial(k, lamb, (N,D))
    # generate target label
    y = np.random.binomial(k, lamb, N)
        
    # if all target labels are identical then flip a label 
    if len(np.unique(y))==1: y[0] = 1-y[0]
    return X, y

def load_linear_data():
    """
    return the dataset for linear SVM
    
    outputs:
        X (np.array): is of shape (N, 2), 2 dimension data
        y (np.array): is of size (N), binary labels with {-1,1}
    
    """
    # Define the dataset
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    X = iris.data[:, :2]
    y = iris.target
    
    kept_index = y<=1
    X = X[kept_index]
    y=y[kept_index]
    y[y==0]=-1
    return X,y

'''Gives two circluar dataset'''
# def load_kernel_dataset_1():
#     from sklearn.datasets import load_iris
#     data = load_iris()
#     X,y = data.data, data.target
#     X_2d = X[:, :2]
#     X_2d = X_2d[y > 0]
#     y_2d = y[y > 0]
#     y_2d -= 1
#     return X_2d,y_2d

def load_kernel_dataset_2():
    np.random.seed(0)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                       X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)
    
    return X_xor, y_xor

def load_kernel_dataset_1():
    np.random.seed(0)
    X,Y = datasets.make_circles(n_samples=100, factor=.5,
                                      noise=.05)
    return X,Y

'''Create data for lda'''
def load_lda_data():
    '''
    return data matrix and num_classes in the dataset.
    '''
    
    iris = datasets.load_iris()
    X = iris.data.astype(np.float32) 
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    labels = iris['target']     
    num_classes = 3
    
    return X[indices,:], labels[indices], num_classes
    
    
    
    
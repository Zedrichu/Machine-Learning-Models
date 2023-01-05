import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.datasets import load_boston

def plotC( fractions, D, list_of_N, num_trails):
    plt.figure()
    domain = np.round(list_of_N)/D    
    plt.plot(domain,fractions)
    
    plt.xlabel("N/D")
    plt.ylabel("C(N,D)")
    
    plt.title("Fraction of convergences per {} trials as a function of N".format(num_trails))
    #print("figure plotted")
    
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

    
    
### Helpers

def vis_classes_prediction(x, labs_gt, labs_pred, C, markers=None, 
                           colors=None, title=None):
    """ Visualizes the data samples from multiple classes and
    shows the difference between GT (markers) and predicted (colors)
    labels.
    
    Args:
        x (np.array): Data samples, shape (N, 2).
        labs_gt (np.array): GT labels, shape (N, ).
        labs_pred (np.array): Pred. labels, shape (N, ).
        C (int): total number of classes.
        markers (list[str]): Markers for each class, len C. If None, 
            automatically selected.
        colors (list[str]): Colors for each class, len C. If None,
            automatically selected.
    """
    N = x.shape[0]
    assert x.ndim == 2 or x.ndim ==3 and np.allclose(x[:, 2], 1.)
    assert labs_gt.shape[0] == N and labs_pred.shape[0] == N
    assert np.min(labs_pred) >= 0 and np.max(labs_pred) < C
    assert labs_gt.ndim <= 2 and labs_pred.ndim <= 2

    # Convert one-hot labels to inds.
    if labs_gt.ndim == 2:
        labs_gt = onehot_to_label(labs_gt)
    if labs_pred.ndim == 2:
        labs_pred = onehot_to_label(labs_pred)
    
    # Strip x of the bias.
    x = x[:, :2]
    
    # Prepare colors and markers.
    markers = markers if markers is not None else ['x', 'o', '+', '*', 'v', '1']
    colors = colors if colors is not None else ['r', 'g', 'b', 'c', 'm', 'y']
    assert len(markers) == len(colors)
    
    if len(markers) != C:
        markers = [markers[i % len(markers)] for i in range(C)]
        colors = [colors[i % len(colors)] for i in range(C)]
        
    markers = np.array(markers)
    colors = np.array(colors)
    
    # Get axes limits.
    xx, yx = np.max(x, axis=0)
    xm, ym = np.min(x, axis=0)
    xrng = xx - xm
    yrng = yx - ym
    sz = 1.1 * np.maximum(xrng, yrng)
    cent = np.array([xm + 0.5 * xrng, ym + 0.5 * yrng])
    xlim = (cent[0] - 0.5 * sz, cent[0] + 0.5 * sz)
    ylim = (cent[1] - 0.5 * sz, cent[1] + 0.5 * sz)
    
    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    for i in range(C):
        inds = np.where(labs_gt == i)[0]
        ax.scatter(*x[inds].T, marker=markers[i], c=colors[labs_pred[inds]])
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if title is not None:
        ax.set_title(title)
        
    return fig

def plot_curve(x1, y1, x2, y2, l1, l2, xl, yl, c1='blue', c2='red', title=''):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.plot(x1, y1, color=c1, label=l1)
    ax.plot(x2, y2, color=c2, label=l2)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title)
    ax.legend()
    
def plot_roc_curves(fpr_W, tpr_W, fpr_V, tpr_V):
    """ Plots ROC curves for 2 classifiers.
    
    Args:
        fpr_W: FP rate for weight classifier, shape (N, ).
        tpr_W: TP rate for weight classifier, shape (N, ).
        fpr_V: FP rate for volume classifier, shape (N, ).
        tpr_V: TP rate for volume classifier, shape (N, ).
    """
    plot_curve(
        fpr_W, tpr_W, fpr_V, tpr_V, 'weight', 'volume', 
        'FP rate', 'TP_rate', title='ROC curve')
    
def plot_prec_rec_curves(recall_W, prec_W, recall_V, prec_V):
    """ Plots Precision Recall curves for 2 classifiers.
    
    Args:
        fpr_W: FP rate for weight classifier, shape (N, ).
        tpr_W: TP rate for weight classifier, shape (N, ).
        fpr_V: FP rate for volume classifier, shape (N, ).
        tpr_V: TP rate for volume classifier, shape (N, ).
    """
    plot_curve(
        recall_W, prec_W, recall_V, prec_V, 'weight', 'volume', 
        'recall', 'precision', title='Precision Recall curve')


### Helpers.

def label_to_onehot(label, C=None):
    C = np.max(label) + 1 if C is None else C
    one_hot_labels = np.zeros(
        (label.shape[0], C), dtype=np.int32)
    one_hot_labels[np.arange(label.shape[0]), label] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    return np.argmax(onehot, axis=1)

def generate_data(mus, stds, Ns, labels_one_hot=True, 
                  bias=False, shuffle=True):
    """ Generates the data from axis-aligned 2D gaussians.
    
    Arguments:
        mus (np.array): Means of each class, shape (C, 2).
        stds (np.array): Std values of the diagonal cov. 
            matrix of each class, shape (C, 2).
        bias (bool): Whether to generate biases (last 
            column of 1s).
        Ns (np.array): Number of samples per class, 
            shape (C, ).
            
    Returns:
        smpls (np.array): Data samples, shape (N, 2).
        labs (np.array): Labels, shape (N, ).
    """
    assert mus.shape == stds.shape
    assert mus.shape[0] == Ns.shape[0]
    C = mus.shape[0]
    
    smpls = np.concatenate([np.random.multivariate_normal(
        mu, np.diag(std), N) for mu, std, N in 
                            zip(mus, stds, Ns)], axis=0)
    labs = np.concatenate([np.ones(
        (N, ), dtype=np.int32) * i for i, N in 
                           enumerate(Ns)], axis=0)
    
    if labels_one_hot:
        labs = label_to_onehot(labs, C=C)
    
    if shuffle:
        rinds = np.random.permutation(smpls.shape[0])
        smpls = smpls[rinds]
        labs = labs[rinds]
        
    if bias:
        smpls = np.concatenate([smpls, np.ones(
            (smpls.shape[0], 1), dtype=np.float32)], axis=1)
    
    return smpls, labs
    

def vis_classes_prediction(x, labs_gt, labs_pred, C, markers=None, 
                           colors=None, title=None):
    """ Visualizes the data samples from multiple classes and
    shows the difference between GT (markers) and predicted (colors)
    labels.
    
    Args:
        x (np.array): Data samples, shape (N, 2).
        labs_gt (np.array): GT labels, shape (N, ).
        labs_pred (np.array): Pred. labels, shape (N, ).
        C (int): total number of classes.
        markers (list[str]): Markers for each class, len C. If None, 
            automatically selected.
        colors (list[str]): Colors for each class, len C. If None,
            automatically selected.
    """
    N = x.shape[0]
    assert x.ndim == 2 or x.ndim ==3 and np.allclose(x[:, 2], 1.)
    assert labs_gt.shape[0] == N and labs_pred.shape[0] == N
    assert np.min(labs_pred) >= 0 and np.max(labs_pred) < C
    assert labs_gt.ndim <= 2 and labs_pred.ndim <= 2

    # Convert one-hot labels to inds.
    if labs_gt.ndim == 2:
        labs_gt = onehot_to_label(labs_gt)
    if labs_pred.ndim == 2:
        labs_pred = onehot_to_label(labs_pred)
    
    # Strip x of the bias.
    x = x[:, :2]
    
    # Prepare colors and markers.
    markers = markers if markers is not None else ['x', 'o', '+', '*', 'v', '1']
    colors = colors if colors is not None else ['r', 'g', 'b', 'c', 'm', 'y']
    assert len(markers) == len(colors)
    
    if len(markers) != C:
        markers = [markers[i % len(markers)] for i in range(C)]
        colors = [colors[i % len(colors)] for i in range(C)]
        
    markers = np.array(markers)
    colors = np.array(colors)
    
    # Get axes limits.
    xx, yx = np.max(x, axis=0)
    xm, ym = np.min(x, axis=0)
    xrng = xx - xm
    yrng = yx - ym
    sz = 1.1 * np.maximum(xrng, yrng)
    cent = np.array([xm + 0.5 * xrng, ym + 0.5 * yrng])
    xlim = (cent[0] - 0.5 * sz, cent[0] + 0.5 * sz)
    ylim = (cent[1] - 0.5 * sz, cent[1] + 0.5 * sz)
    
    # Plot.
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    for i in range(C):
        inds = np.where(labs_gt == i)[0]
        ax.scatter(*x[inds].T, marker=markers[i], c=colors[labs_pred[inds]])
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    if title is not None:
        ax.set_title(title)
        
    return fig


def generate_apple_grenade_data():
    np.random.seed(10)
    
    ### Generate data.
    muV_apple, stdV_apple = 0.18, 0.005  # liters
    muV_gran, stdV_gran = 0.35, 0.015
    muW_apple, stdW_apple = 0.085, 0.015  # kg
    muW_gran, stdW_gran = 0.380, 0.045
    N_apple, N_gran = 1000, 20
    N = N_apple + N_gran

    rinds = np.random.permutation(N)
    x_apple = np.maximum(np.random.multivariate_normal(
        (muW_apple, muV_apple), np.diag((stdW_apple, stdV_apple)), N_apple),
                        np.array([0.03, 0.01]))
    x_gran = np.maximum(np.random.multivariate_normal(
        (muW_gran, muV_gran), np.diag((stdW_gran, stdV_gran)), N_gran),
                    np.array([0.08, 0.02]))
    X = np.concatenate([x_apple, x_gran], axis=0)[rinds]
    y = np.concatenate([np.zeros((N_apple, )), np.ones((N_gran, ))], 
                        axis=0)[rinds].astype(np.int32)

    return X, y

def plot_apple_grenade_data(X, y, class_thresh_func):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(*X[y == 0].T, marker='o', color='g', label='apples')
    ax.scatter(*X[y == 1].T, marker='x', color='r', label='granades')
    ax.set_xlabel('weight [kg]')
    ax.set_ylabel('volume [l]')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
    plt.close(fig)

    # Handpicked thresholds.
    tW = 0.15
    tV = 0.2

    yW = class_thresh_func(X[:, 0], tW)
    yV = class_thresh_func(X[:, 1], tV)

    _ = vis_classes_prediction(
        X, y, yW, 2, markers=['o', 'x'], colors=['g', 'r'], 
        title='Weight classifier')
        
    _ = vis_classes_prediction(
        X, y, yV, 2, markers=['o', 'x'], colors=['g', 'r'], 
        title='Volume classifier')
    

def load_boston_dataset():
    # get the data set and print a description
    boston_dataset = load_boston()
    print(boston_dataset.DESCR)

    X = boston_dataset["data"]
    y = boston_dataset["target"]

    # remove categorical feature
    X = np.delete(X, 3, axis=1)
    # removing second mode
    ind = y<40
    X = X[ind,:]
    y = y[ind]

    # split the data into 80% training and 20% test data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    splitRatio = 0.8
    n          = X.shape[0]
    X_train    = X[indices[0:int(n*splitRatio)],:] 
    y_train    = y[indices[0:int(n*splitRatio)]] 
    X_test     = X[indices[int(n*(splitRatio)):],:] 
    y_test     = y[indices[int(n*(splitRatio)):]] 

    return X_train, y_train, X_test, y_test

def load_knn_data():
    # Paths
    features_annotated_path = "solutions/data/feats_annotated.npy"     # Weights, heights of individuals with known body category
    labels_annotated_path = "solutions/data/labels_annotated.npy"      # Body categories of those individuals
    features_unannotated_path = "solutions/data/feats_unannotated.npy" # Weights and heights of unknown body category individuals
    labels_unannotated_path = "solutions/data/labels_unannotated_secret.npy"     # - Goal: Figure out their body categories

    # Features organized in an NxD matrix; N examples, D features.
    # Another way to look at it: each of the N examples is a D-dimensional feature vector.

    data_train = np.load(features_annotated_path)
    data_test = np.load(features_unannotated_path)
    labels_train = np.load(labels_annotated_path)
    labels_test = np.load(labels_unannotated_path)

    class_names = ["Class0", "Class1"]
    return data_train, data_test, labels_train, labels_test, class_names

def plot_knn_training_test(data_train, data_test, labels_train, labels_test, colors, class_names):
    # Visualize training and test sets
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    plt.title(f"Training set ({len(labels_train)} examples)")
    for i, class_name in enumerate(class_names):
        plt.scatter(*data_train[labels_train==i].T,c=colors[i, None], alpha=0.5, s=15, lw=0, label=class_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend();
    plt.show()
    plt.close()

    plt.subplot(1,2,2)
    plt.title(f"Test set ({len(labels_test)} examples)")
    for i, class_name in enumerate(class_names):
        plt.scatter(*data_test[labels_test==i].T,
                    c=colors[i, None], alpha=0.5, s=15, lw=0, label=class_name)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend();
    plt.show()
    plt.close()

def my_accuracy_func(pred, gt):
    return np.mean(pred==gt)
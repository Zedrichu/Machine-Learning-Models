import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
from mpl_toolkits.mplot3d import axes3d, Axes3D 



def generate_data_kernel():
    Ns = 400
    std = 0.02
    smpls = np.random.multivariate_normal((0,0,0), np.diag((std,std,std)), Ns)

    mean_radii = [0.6, 1]
    for rad in zip(mean_radii):
        r = np.random.normal(rad, std, Ns)

        phi = np.linspace(0, 2*np.pi, int(np.sqrt(Ns))) + np.random.normal(0, 0.1, int(np.sqrt(Ns)))
        theta = np.linspace(0, np.pi, int(np.sqrt(Ns))) + np.random.normal(0, 0.1, int(np.sqrt(Ns)))

        theta, phi = np.meshgrid(theta, phi)

        r_xy = r*np.sin(theta.flatten())
        x = np.cos(phi.flatten()) * r_xy
        y = np.sin(phi.flatten()) * r_xy
        z = r * np.cos(theta.flatten())
        smpls = np.concatenate((smpls, np.concatenate([x[:,np.newaxis],y[:,np.newaxis],z[:,np.newaxis]], axis=1)), axis=0)

    labs = np.concatenate([np.ones(
        (N, ), dtype=np.int32) * (i-1) for i, N in 
                        enumerate([Ns, Ns, Ns])], axis=0)

    #labs = label_to_onehot(labs, C=3)

    rinds = np.random.permutation(smpls.shape[0])
    smpls = smpls[rinds]
    labs = labs[rinds]
        
    return smpls, labs
    
def label_to_onehot(label, C=None):
    C = np.max(label) + 1 if C is None else C
    one_hot_labels = np.zeros(
        (label.shape[0], C), dtype=np.int32)
    one_hot_labels[np.arange(label.shape[0]), label+1] = 1
    return one_hot_labels

def onehot_to_label(onehot):
    return np.argmax(onehot, axis=1)-1

def vis_data_kernel(X, Y, title_text):
    if Y.ndim>1:
        labs = onehot_to_label(Y)
    else:
        labs = Y

    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ["r", "b", "g"]
    for i in range(-1,2):
        ax.scatter(X[labs==i,0],X[labs==i,1],X[labs==i,2], color=colors[i])
        ax.set_title(title_text)
    return fig


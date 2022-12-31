import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from mlxtend.plotting import plot_decision_regions

COLORS = ['orange', 'blue']

def plot_two_classes(X, y):
    '''
    Plots two-class two-dimensional data distribution X given binary labels y.
    '''
    mask = (y == 1)
    
    fontsize = 12
    _, ax = plt.subplots(figsize=(10,5))
    ax.scatter(X[~mask, 0], X[~mask, 1], color=COLORS[0], label='Class 0', alpha=0.5, s=10)
    ax.scatter(X[mask, 0], X[mask, 1], color=COLORS[1], label='Class 1', alpha=0.5, s=10)
    ax.set_xlabel('Dim 1', fontsize=fontsize)
    ax.set_ylabel('Dim 2', fontsize=fontsize)
    _ = ax.legend(loc='best', fontsize=fontsize)


def plot_statistics_curves(X, y, clf):

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    ### plot classes with decision boundary
    _ = plot_decision_regions(X=X, y=y, clf=clf, legend=2, colors='orange,darkblue', ax=ax[0])

    y_probas = clf.decision_function(X)

    # plot scores across thresholds
    fpr, tpr, thresholds = roc_curve(y, y_probas, pos_label=1)
    AUROC = auc(fpr, tpr)
    ax[1].plot(fpr, tpr)
    ax[1].set_xlabel('FPR', fontsize=16)
    ax[1].set_ylabel('TPR', fontsize=16)
    ax[1].set_title(f'AUC ROC curve, AUC={AUROC:0.2f}')

    AP = average_precision_score(y, y_probas)
    pr, rc, thresholds = precision_recall_curve(y, y_probas)
    ax[2].plot(pr, rc)
    ax[2].set_xlabel('Precision', fontsize=16)
    ax[2].set_ylabel('Recall', fontsize=16)
    ax[2].set_title(f'Precision-Recall curve: AP={AP:0.2f}')
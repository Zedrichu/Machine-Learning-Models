import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNN(object):
    """
        kNN classifier object. 
        
        Note: It is implemented using Scikit-Learn, which you are NOT allowed to use for the project!
    """

    def __init__(self, k=1):
        self.task_kind = 'classification'
        self.set_arguments(k=k)

    def set_arguments(self, k=1):
        self.k=k
        self.knn = KNeighborsClassifier(self.k)
    
    def fit(self, training_data, training_labels):
        self.knn.fit(training_data, training_labels)
        return self.knn.predict(training_data)
                               
    def predict(self, test_data):
        return self.knn.predict(test_data)

def mse_fn(pred,gt):
    '''
        Mean Squared Error
        Arguments:
            pred: NxD prediction matrix
            gt: NxD groundtruth values for each predictions
        Returns:
            returns the computed loss
    '''
    loss = (pred-gt)**2
    loss = np.mean(loss)
    return loss
    

def macrof1_fn(pred_labels,gt_labels):
    '''
        Macro F1 score
        Arguments:
            pred_labels: N prediction labels
            gt_labels: N corresponding gt labels
        Returns:
            returns the computed macro f1 score
    '''
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)
        macrof1 += 2*(precision*recall)/(precision+recall)
    return macrof1/len(class_ids)
import numpy as np    
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid



def get_w_analytical(X_train,y_train):
    """
    compute the weight parameters w
    """
    
    """
    Please fill in the required code here
    """
        
    # compute w via the normal equation
    # np.linalg.solve is more stable than np.linalg.inv
    w = np.linalg.solve(X_train.T@X_train,X_train.T@y_train)
    return w

def get_loss(w, X_train, y_train,X_test,y_test,val=False):
    # predict dependent variables and MSE loss for seen training data
    """
    Please fill in the required code here
    """
    loss_train = (np.mean((y_train-X_train@w)**2))
    loss_train_std = np.std((y_train-X_train@w)**2)
    
    # predict dependent variables and MSE loss for unseen test data
    """
    Please fill in the required code here
    """
    loss_test = (np.mean((y_test-X_test@w)**2))
    loss_test_std = np.std((y_test-X_test@w)**2)
    if not val:
        print("The training loss is {} with std:{}. The test loss is {} with std:{}.".format(loss_train, loss_train_std, loss_test,loss_test_std))
    else:
        print("The training loss is {} with std:{}. The val loss is {} with std:{}.".format(loss_train, loss_train_std, loss_test,loss_test_std))

    return loss_test


def expand_X(X,d):
    """
    perform degree-d polynomial feature expansion of X, with bias but omitting interaction terms
    """
    expand = np.ones((X.shape[0],1))
    for idx in range(1,d+1): expand=np.hstack((expand, X**idx))
    return expand


# Function for using kth split as validation set to get loss
# and k-1 splits to train our model
# k = kth fold
# k_fold_ind = all the fold indices
# X,Y= train data and labels
# degree = degree of polynomial expansion
def do_cross_validation(k,k_fold_ind,X,Y, expand_fn, expand_and_normalize_X, degree=1):
    # use one split as val
    val_ind = k_fold_ind[k]
    # use k-1 split to train
    train_splits = [i for i in range(k_fold_ind.shape[0]) if i is not k]
    train_ind = k_fold_ind[train_splits,:].reshape(-1)
    
    #Get train and val 
    cv_X_train = X[train_ind,:]
    cv_Y_train = Y[train_ind]
    cv_X_val = X[val_ind,:]
    cv_Y_val = Y[val_ind]

    #expand and normalize for degree d
    cv_X_train_poly,mu,std = expand_and_normalize_X(cv_X_train, degree, expand_fn)
    #apply the normalization using statistics (mean, std) computed on train data
    cv_X_val_poly = expand_fn(cv_X_val,degree)
    cv_X_val_poly[:,1:] =  (cv_X_val_poly[:,1:]-mu)/std
    
    #fit on train set
    w = get_w_analytical(cv_X_train_poly,cv_Y_train)
    
    #get loss for val
    loss_test = get_loss(w,cv_X_train_poly,cv_Y_train,cv_X_val_poly,cv_Y_val,val=True)
    return loss_test



# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
def fold_indices(num_examples, k_fold):
    # try to load first
    path_grade_data = 'data/linear_expansion_with_cross_validation.npz'
    with np.load(path_grade_data, allow_pickle=True) as data_file:
        fold_indices = dict(data_file.items())['fold_indices']
        if fold_indices.shape[0] == k_fold:
            print('Load the fold indices successfully!')
            return fold_indices
        
    print('WARNING: The shape of num_examples and k_fold does not match the pre-computed indices. Generate from scratch.')
    # generate if the shape is wrong ...
    ind = np.arange(num_examples)
    split_size = num_examples//k_fold
    np.random.seed(0)
    #important to shuffle your data
    np.random.shuffle(ind)
    k_fold_indices = []
    # Generate k_fold set of indices
    k_fold_indices = [ind[k*split_size:(k+1)*split_size] for k in range(k_fold)]
    return np.array(k_fold_indices)


'''Plotting Heatmap for CV results'''
def plot_cv_result(grid_val,grid_search_lambda,grid_search_degree):
    plt.figure(figsize=(8,10))
    plt.imshow(grid_val)
    plt.colorbar()
    plt.xticks(np.arange(len(grid_search_degree)), grid_search_degree, rotation=20)
    plt.yticks(np.arange(len(grid_search_lambda)), grid_search_lambda, rotation=20)
    plt.xlabel('degree')
    plt.ylabel('lambda')
    plt.title('Val Loss for different lambda and degree')
    plt.show()
    

'''
Grid Search Function
params:{'param1':[1,2,..,4],'param2':[6,7]} dictionary of search params
k_fold: fold for CV to be done
fold_ind: splits of training set
function: implementation of model should return a loss or score
expand_fn: expansion function that is passed into function
X,Y: training examples
'''
def grid_search_cv(params,k_fold,fold_ind,expand_fn,expand_and_normalize,X,Y):
    
    #might mess up with dictionary order
    param_grid = ParameterGrid(params)
    #save the values for the combination of hyperparameters
    grid_val = np.zeros(len(param_grid))
    grid_val_std = np.zeros(len(param_grid))   
    
    for i, p in enumerate(param_grid):
        print('Evaluating for {} ...'.format(p))
        loss = np.zeros(k_fold)
        for k in range(k_fold):
            loss[k] = do_cross_validation(k,fold_ind,X,Y,expand_fn,expand_and_normalize,**p)
        grid_val[i] = np.mean(loss)
        grid_val_std[i] = np.std(loss)
    
    # reshape in the proper dimension of search space
    if len(params.keys())>1:
        search_dim = tuple([len(p) for _,p in params.items()])
        grid_val = grid_val.reshape(search_dim)
        grid_val_std = grid_val_std.reshape(search_dim)
    
    return grid_val, grid_val_std
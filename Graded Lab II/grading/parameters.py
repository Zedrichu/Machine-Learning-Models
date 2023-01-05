""" Parameters for generating grading data.
"""

grading_params = {
    'kernel_function': dict(D=5),
    'find_kernel_matrices': dict(N_train=500, N_test=100, D=5),
    'empirical_risk': dict(N_train=500),
    'find_alpha_star': dict(N_train=500),
    'prediction': dict(N_train=500, N_test=100),
    'thresholding': dict(N_test=100),
    'pca': None
}

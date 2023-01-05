import numpy as np
from matplotlib import pyplot as plt
import grading.helpers as helpers


################################################################################
# Saving student results
################################################################################
def initialize_res(scope):
    exercise_id = "sciper"
    sciper_number = helpers.resolve('sciper_number', scope)
    stud_grad = dict(sciper_number=sciper_number)
    helpers.register_answer(exercise_id, stud_grad, scope)

def kernel_mcq(scope):
    exercise_id = "kernel_mcq"
    ans = helpers.resolve('kernel_mcq', scope)
    stud_grad = dict(kernel_mcq=ans)
    helpers.register_answer(exercise_id, stud_grad, scope)

def kernel_function(scope):
    exercise_id = 'kernel_function'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res = func(test_data['gr_X1'],test_data['gr_X2'])
    
    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)

def find_kernel_matrices(scope):

    # kernel function to generate the grading solutions
    def gr_kernel_func(x_i, x_j):
        k = 1 + x_i@x_j
        return k

    exercise_id = 'find_kernel_matrices'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res1, stud_res2 = func(test_data['X_train'],test_data['X_test'],gr_kernel_func)
    
    #save the students' results
    stud_grad = dict(stud_res1=stud_res1,stud_res2=stud_res2)
    helpers.register_answer(exercise_id, stud_grad, scope)    

def empirical_risk(scope): 
    exercise_id = 'empirical_risk'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res = func(test_data['K_train'],test_data['Y_train'],test_data['alpha'])
    
    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)

def find_alpha_star(scope):
    exercise_id = 'find_alpha_star'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res = func(test_data['K_train'],test_data['Y_train'])
    
    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)
    
def prediction(scope):
    exercise_id = 'prediction'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res = func(test_data['alpha'],test_data['K_test'])
    
    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)
    
def thresholding(scope):
    exercise_id = 'thresholding'
    
    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)
    
    # Get the grading data
    test_data = helpers.get_data(exercise_id)
    
    # run the students' functions
    stud_res = func(test_data['raw_pred'])
    
    #save the students' results
    stud_grad = dict(stud_res=stud_res)
    helpers.register_answer(exercise_id, stud_grad, scope)

def pca(scope):
    exercise_id = 'pca'

    # Get functions for which to generate test and grading data.
    func = helpers.resolve(exercise_id, scope)

    # Get grading data
    test_data = helpers.get_data(exercise_id)

    student_results = {}
    # run student function
    for k in test_data.keys():
        student_results[k] = func(test_data[k], int(k))
    
    helpers.register_answer(exercise_id, student_results, scope)

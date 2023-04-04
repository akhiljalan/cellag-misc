import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy 

def rescale_according_to_target_mat(target_mat, mat_to_rescale):
    target_average_degree = np.mean(np.sum(target_mat, axis=1))
    mat_to_rescale_average_degree = np.mean(np.sum(mat_to_rescale, axis=1))
    scale = target_average_degree / mat_to_rescale_average_degree
    return mat_to_rescale * scale


def gen_sample_mat(ground_truth_mat):
    '''
    Generate a sample matrix from a ground truth matrix
    inputs:
        ground_truth_mat: A n by n matrix of probabilities
    outputs:    
        A: a n by n matrix whose entries are 1 
        with probability ground_truth_mat[i, j] and 0 otherwise
    '''
    n = ground_truth_mat.shape[0]
    A = np.zeros_like(ground_truth_mat)
    for i in range(n):
        for j in range(i + 1):
            sample = np.random.binomial(1, ground_truth_mat[i, j])
            A[i, j] = sample
            A[j, i] = sample
    return rescale_according_to_target_mat(ground_truth_mat, A)

def gen_sample_mat_normal_distr(ground_truth_mat): 
    n = ground_truth_mat.shape[0]
    A = np.zeros_like(ground_truth_mat)
    for i in range(n):
        for j in range(i + 1):
            sample = np.random.normal(ground_truth_mat[i, j], 0.1)
            A[i, j] = sample
            A[j, i] = sample
    return rescale_according_to_target_mat(ground_truth_mat, A)

def generate_ground_truth_mat(n): 
    '''
    Generate a symmetric ground truth matrix with n rows and columns
    and uniform(0, 1) each entry
    '''
    ground_truth_asymm = np.random.uniform(0, 0.5, (n, n))
    ground_truth_mat = ground_truth_asymm + ground_truth_asymm.T
    return ground_truth_mat

def generate_samples_from_ground_truth(ground_truth_mat, num_samples): 
    '''
    Generate sample matrices from a ground truth matrix
    '''
    sample_mats = [gen_sample_mat(ground_truth_mat) for _ in range(num_samples)]
    return sample_mats

def generate_synthetic_data(n, num_sample_mats):  
    '''
    n: number of rows and columns in the ground truth matrix
    num_sample_mats: number of sample matrices to generate
    '''
    ground_truth_asymm = np.random.uniform(0, 0.5, (n, n))
    ground_truth_mat = ground_truth_asymm + ground_truth_asymm.T

    sample_mats = [gen_sample_mat(ground_truth_mat) for _ in range(num_sample_mats)]

    scale = np.sum(sample_mats[0])
    for i in range(len(sample_mats)):
        if i > 0:
            s = np.sum(sample_mats[i])
            rescale = scale / s 
            sample_mats[i] = rescale * sample_mats[i]
    return sample_mats 

def matrix_lin_combo(eta_arr, sample_mats): 
    '''
    eta_arr: array of weights for linear combination
    sample_mats: array of sample matrices

    returns: linear combination of the form
    eta_0 * sample_mats[0] + sum_{i > 0} eta_i * (X_0 - X_i)
    '''
    X_0 = np.copy(eta_arr[0] * sample_mats[0])
    for i in range(1, len(sample_mats)): 
        X_i = eta_arr[i] * (X_0 - sample_mats[i])
        X_0 += X_i
    return X_0

def matrix_lin_combo_pos_sign(eta_arr, sample_mats):  
    return np.sum([eta * sample_mat for eta, sample_mat in zip(eta_arr, sample_mats)], axis=0)

def get_delta(X_0): 
    average_degree = np.mean(np.sum(X_0, axis=0))
    return np.sqrt(average_degree)

def simplex_constraint(eta_arr): 
    return np.sum(eta_arr) - 1.0

def generate_random_eta(m): 
    eta_init = np.random.uniform(0, 1, m)
    eta_init = eta_init / np.sum(eta_init)
    return eta_init

def shift_scale_to_zero_one(mat_2d, return_verbose=False): 
    min_val = np.min(mat_2d)
    max_val = np.max(mat_2d)
    scale = 1.0 / (max_val - min_val)
    shift = -min_val * scale
    if return_verbose:
        return mat_2d * scale + shift, scale, shift
    return mat_2d * scale + shift
    # return (mat_2d - min_val) / (max_val - min_val)
    

def objective_with_params(eta_arr, validation_mat, sample_mats, delta, num_eigs_included=None, verbose=False):
    if verbose:
        print(eta_arr)
    if num_eigs_included is None:
        num_eigs_included = validation_mat.shape[0]
    P_hat = matrix_lin_combo_pos_sign(eta_arr, sample_mats)

    # singular values, decreasing order
    diff_sing_values = scipy.linalg.svdvals(validation_mat - P_hat)

    def ob_fn(k): 
        return sum(diff_sing_values[:k]) - k * delta

    all_obj_values = [ob_fn(k) for k in range(1, num_eigs_included + 1)]
    max_obj_index = np.argmax(all_obj_values)

    return all_obj_values[max_obj_index]

def objective_with_params_scale_x0(eta_arr, validation_mat, 
        sample_mats, delta, num_eigs_included=None, verbose=False):
    if verbose:
        print(eta_arr)
    if num_eigs_included is None:
        num_eigs_included = validation_mat.shape[0]

    # eta_1, ..., eta_m are the weights for the P_i
    P_hat = matrix_lin_combo_pos_sign(eta_arr[1:], sample_mats)

    # singular values, decreasing order
    diff_sing_values = scipy.linalg.svdvals(eta_arr[0] * validation_mat - P_hat)

    def ob_fn(k): 
        return sum(diff_sing_values[:k]) - k * delta

    all_obj_values = [ob_fn(k) for k in range(1, num_eigs_included + 1)]
    max_obj_index = np.argmax(all_obj_values)

    return all_obj_values[max_obj_index]

def run_scipy_minimize(eta_init, sample_mats, validation_mat, delta, num_eigs_included=None, verbose=False):
    
    objective = lambda eta_arr: objective_with_params(eta_arr, validation_mat, sample_mats, delta, verbose=False)

    result = scipy.optimize.minimize(
        objective,
        eta_init,
        method='SLSQP',
        jac=None,
        bounds=[(0, 1) for _ in range(len(sample_mats))],
        constraints={'type': 'eq', 'fun': simplex_constraint},
        options={
            'maxiter': 10000,
            'disp': False
        }
    )
    return result

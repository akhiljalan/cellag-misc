import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy 
import fbpca

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
    tol = 1e-6
    n = ground_truth_mat.shape[0]
    A = np.zeros_like(ground_truth_mat)
    for i in range(n):
        for j in range(i + 1):
            if ground_truth_mat[i, j] < tol: 
                A[i, j] = 0.0
                A[j, i] = 0.0
            sample = np.random.binomial(1, ground_truth_mat[i, j])
            A[i, j] = sample
            A[j, i] = sample
        # print(i)
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

def matrix_lin_combo_pos_sign(eta_arr, sample_mats, sparse=False):  
    if sparse: 
        return np.sum([eta_arr[i] * sample_mats[i] for i in range(len(eta_arr))])
        return scipy.sparse.csr_matrix.sum([eta * sample_mat for eta, sample_mat 
                                            in zip(eta_arr, sample_mats)], axis=0)
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

def objective_with_params_sparse(eta_arr, validation_mat, 
        sample_mats, delta, num_eigs_included=None, verbose=False, use_random_svd=True):
    # assumes that all input matrices are scipy sparse format
    if num_eigs_included is None:
        num_eigs_included = validation_mat.shape[0]
    P_hat = matrix_lin_combo_pos_sign(eta_arr, sample_mats, sparse=True)

    # singular values, increasing order
    if use_random_svd: 
        U, S, Vh = fbpca.pca(validation_mat - P_hat, k = num_eigs_included, n_iter=4)
        diff_sing_values = S
    else: 
        sing_vals = scipy.sparse.linalg.svds(validation_mat - P_hat, solver='arpack',
                                                k=num_eigs_included - 1, return_singular_vectors=False)
        diff_sing_values = np.flip(sing_vals)

    # assumes sing values are in decrasing order
    def ob_fn(k): 
        return sum(diff_sing_values[k:]) - k * delta

    all_obj_values = [ob_fn(k) for k in range(1, num_eigs_included + 1)]
    max_obj_index = np.argmax(all_obj_values)
    return all_obj_values[max_obj_index]

def run_scipy_minimize(sample_mats, validation_mat, delta, eta_init=None, num_eigs_included=None, verbose=False):

    if eta_init is None: 
        eta_init = generate_random_eta(len(sample_mats))
    objective = lambda eta_arr: objective_with_params(eta_arr, validation_mat, sample_mats, delta, verbose=False)
    print('begin iter')
    # constraints={'type': 'eq', 'fun': simplex_constraint},
    result = scipy.optimize.minimize(
        objective,
        eta_init,
        method='SLSQP',
        jac=None,
        bounds=[(0, 1) for _ in range(len(sample_mats))],
        options={
            'maxiter': 10000,
            'disp': verbose
        }
    )
    return result

def delta_estimate(validation_mat, scaling=1.0): 
    max_degree = np.max(np.sum(validation_mat, axis=0))
    return scaling * np.sqrt(max_degree)

def gen_occluded_p(sparse_M, frac_to_occlude = 0.01): 
    n = sparse_M.shape[0]
    num_to_occlude = int(frac_to_occlude * n)
    occluded_indices = np.random.choice(n, size=num_to_occlude, replace=False)
    return zero_rows_cols(sparse_M, occluded_indices)

def zero_rows_cols(M, row_indices):
    '''
    Zeroes out the rows/cols in row_indices. 
    M is a sparse matrix type. 
    '''
    diag = scipy.sparse.eye(M.shape[0]).tolil()
    for r in row_indices:
        diag[r, r] = 0
    return diag.dot(M).dot(diag)

def bin_samples_rand2(n, mu):
    '''
    mu: List of floats in [0,1] describing Bernoulli mean probabilities. 
    n: number of samples per entries of mu. 
    '''
    rng = np.random.default_rng(seed=None)
    return (rng.random(size=(len(mu), n)) < mu[:, None]).astype(np.uint8)

def gen_sparse_sample_boolean_mat(sparse_mat): 
    nonzero_float_entries = scipy.sparse.triu(sparse_mat, k=1).data
    sparse_tri = scipy.sparse.csr_matrix(scipy.sparse.triu(sparse_mat, k = 1))
    sample_bool = bin_samples_rand2(1, nonzero_float_entries).squeeze(-1)
    sparse_tri.data = sample_bool
    symm_sparse = sparse_tri + sparse_tri.T
    return symm_sparse

def result_diff(result_dict, sample_mats, validation_mat, ground_truth_mat, num_eigs_included): 
    P_hat = matrix_lin_combo_pos_sign(result_dict['x'], sample_mats)
    validation_diff_svals = scipy.sparse.linalg.svds(validation_mat - P_hat, solver='arpack',
                                                k=num_eigs_included - 1, return_singular_vectors=False)
    true_diff_svals = scipy.sparse.linalg.svds(ground_truth_mat - P_hat, solver='arpack',
                                                k=num_eigs_included - 1, return_singular_vectors=False)
    delta_est = delta_estimate(validation_mat)
    true_delta = scipy.sparse.linalg.norm(validation_mat - ground_truth_mat, ord=2)
    return validation_diff_svals, true_diff_svals, delta_est, true_delta

def plot_scipy_optimize_result_and_save(result_dict, sample_mats, validation_mat, ground_truth_matrix, 
                         num_eigs_solver, num_eigs_to_show, occlusion_level=0.01, m=5, 
                         savepath=None): 
    validation_diff_svals, true_diff_svals, delta_est, true_delta = result_diff(result_dict, sample_mats, 
        validation_mat, ground_truth_matrix, num_eigs_to_show)


    plt.bar(np.arange(len(true_diff_svals)), sorted(true_diff_svals, reverse=True), 
            label='True Diff', width=0.3)
    plt.bar(np.arange(len(true_diff_svals)) + 0.5, sorted(validation_diff_svals, reverse=True), 
            label='Validation Diff', width=0.3)
    plt.xlabel('Index')
    plt.ylabel('Singular value')
    plt.axhline(delta_est, ls='--', color='green', label='$\sqrt{\max_i \sum_j A_{ij}^{(s)} }$')
    plt.axhline(true_delta, ls='--', color='blue', label='$\| A^{(s)} - P \|_2$')
    plt.axvline(num_eigs_solver + 0.8, ls='--', color='black')
    plt.legend()
    plt.tight_layout()
    plt.title('Solving for {} Singular Vals, {} pct occlusion, Human PPI, m={}'.format(num_eigs_solver, 100 * occlusion_level, m))
    if savepath is not None: 
        plt.savefig(savepath, dpi=400.0)
from datetime import datetime
import numpy as np
import sys

# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import networkx as nx
import scipy
# import pickle
# import fbpca
import graph_learning_utils as gl
# plt.rcParams['figure.figsize'] = [15, 10]
# sns.set_style('whitegrid')
# plt.rcParams['font.size'] = 20.0
# plt.rcParams['xtick.labelsize'] = 20.0
# plt.rcParams['ytick.labelsize'] = 20.0
# import pickle
from itertools import product
import pandas as pd

def get_cur_time_str(): 
    now = datetime.now() # current date and time
    # date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    return now

def load_human_ppi(): 
    human_ppi_mat = scipy.sparse.load_npz('data/adj_matrix_sparse_restricted_9606.npz')
    max_degree =  np.max(human_ppi_mat)
    human_ppi_mat /= max_degree
    return human_ppi_mat

def gen_sample_data(base_ppi_mat, m=5, occlusion_rate=0.01, delta_scaling = 0.1): 
    occluded_ground_truths = [gl.gen_occluded_p(base_ppi_mat, frac_to_occlude=occlusion_rate) for _ in range(m)]
    validation_mat = gl.gen_sparse_sample_boolean_mat(base_ppi_mat)
    sample_mats = [gl.gen_sparse_sample_boolean_mat(occluded_ground_truth) for occluded_ground_truth in occluded_ground_truths]
    eta_init = gl.generate_random_eta(m)
    delta = gl.delta_estimate(validation_mat, scaling=delta_scaling)
    return validation_mat, sample_mats, eta_init, delta

def print_ln(): 
    print('--------------------------\n')

def main(out_path): 
    out_df = pd.DataFrame()
    cur_time = get_cur_time_str()

    print('BEGIN TEST at time {}'.format(cur_time))
    human_ppi_mat = load_human_ppi()
    num_eigs_to_include = 10

    m_range = [5, 10, 15, 20, 30]
    delta_scale_range = [0.0, 0.1] #exclude 0.5
    occlusion_levels = [0.0, 0.1, 0.5]
    # k_levels = [10, 20, 30]
    num_eigs_to_test = 20

    param_sets = product(m_range, delta_scale_range, occlusion_levels)
    for m_val, delta_val, occ_val in param_sets: 
        cur_data = {
            'm': m_val, 
            'delta_scaling': delta_val, 
            'occlusion_level': occ_val
        }
        print_ln()
        begin_time = get_cur_time_str()
        print('BEGIN TEST at time {}'.format(begin_time))
        print('Running test with m={}, delta_scaling={}, occlusion level={}, k={}'.format(m_val, 
                                                                                          delta_val, occ_val,num_eigs_to_include))

        validation_mat, sample_mats, eta_init, delta = gen_sample_data(human_ppi_mat, m=m_val, 
            occlusion_rate=occ_val, delta_scaling=delta_val)

        objective = lambda eta_arr: gl.objective_with_params_sparse(eta_arr, validation_mat, 
                                                                    sample_mats, delta, 
                                                                    num_eigs_included = num_eigs_to_include, 
                                                                    use_random_svd=False)
        bfgs_options = {
            'iprint': 5, 
            'maxiter': 500, 
        }

        result = scipy.optimize.minimize(
            objective,
            eta_init,
            method='L-BFGS-B',
            jac=None,
            bounds=[(0, 1) for _ in range(len(sample_mats))],
            options=bfgs_options
        )
        # print(result)
        validation_diff_svals, true_diff_svals, delta_est, true_delta = gl.result_diff(result, sample_mats, 
            validation_mat, human_ppi_mat, num_eigs_included=num_eigs_to_test)
        # print('L2 diff of validation and ground truth {}'.format(true_delta))
        # print('Top {} validation diff singular values {}'.format(num_eigs_to_test, validation_diff_svals))
        # print('Top {} v true diff singular values {}'.format(num_eigs_to_test, true_diff_svals))

        end_time = get_cur_time_str()
        print('END TEST at time {}'.format(end_time))
        print_ln()
        time_diff = (end_time - begin_time).total_seconds()

        cur_data['l2_diff_val_gt'] = true_delta
        cur_data['validation_diff_svals'] = validation_diff_svals
        cur_data['true_diff_svals'] = true_diff_svals
        cur_data['time_diff'] = time_diff
        cur_data.update(result)

        out_df = out_df.append(cur_data, ignore_index=True)
        out_df.to_csv(out_path)

if __name__ == '__main__': 
    now = datetime.now()
    out_path = 'logs/job_{}.csv'.format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    print('Writing to {}'.format(out_path))
    main(out_path=out_path)
from datetime import datetime
import numpy as np
import sys

import matplotlib.pyplot as plt
import seaborn as sns
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

def gen_sample_data(base_ppi_mat, m=5, occlusion_rate=0.01, delta_scaling = 0.1): 
    occluded_ground_truths = [gl.gen_occluded_p(base_ppi_mat, frac_to_occlude=occlusion_rate) for _ in range(m)]
    validation_mat = gl.gen_sparse_sample_boolean_mat(base_ppi_mat)
    sample_mats = [gl.gen_sparse_sample_boolean_mat(occluded_ground_truth) for occluded_ground_truth in occluded_ground_truths]
    eta_init = gl.generate_random_eta(m)
    delta = gl.delta_estimate(validation_mat, scaling=delta_scaling)
    return validation_mat, sample_mats, eta_init, delta

def print_ln(): 
    print('--------------------------\n')

def run_tests_one_ground_truth(ground_truth_mat,
         m_range=[5, 10, 15, 20, 30], 
         delta_scale_range=[0.0, 0.1],
         occlusion_levels=[0.0, 0.1, 0.5],
         num_eigs_to_test=20,
         bfgs_options = {
            'iprint': 5, 
            'maxiter': 500, 
        }, 
        out_path=None): 
    out_df = pd.DataFrame()
    cur_time = get_cur_time_str()

    print('BEGIN TEST at time {}'.format(cur_time))
    # human_ppi_mat = load_human_ppi()
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
                                                                                          delta_val, 
                                                                                          occ_val,num_eigs_to_include))

        validation_mat, sample_mats, eta_init, delta = gen_sample_data(ground_truth_mat, m=m_val, 
            occlusion_rate=occ_val, delta_scaling=delta_val)

        objective = lambda eta_arr: gl.objective_with_params_sparse(eta_arr, validation_mat, 
                                                                    sample_mats, delta, 
                                                                    num_eigs_included = num_eigs_to_include, 
                                                                    use_random_svd=False)
        # bfgs_options = {
        #     'iprint': 5, 
        #     'maxiter': 500, 
        # }

        result = scipy.optimize.minimize(
            objective,
            eta_init,
            method='L-BFGS-B',
            jac=None,
            bounds=[(0, 1) for _ in range(len(sample_mats))],
            options=bfgs_options
        )
        validation_diff_svals, true_diff_svals, delta_est, true_delta = gl.result_diff(result, sample_mats, 
            validation_mat, ground_truth_mat, num_eigs_included=num_eigs_to_test)

        end_time = get_cur_time_str()
        print('END TEST at time {}'.format(end_time))
        print_ln()
        time_diff = (end_time - begin_time).total_seconds()

        cur_data['l2_diff_val_gt'] = true_delta
        cur_data['delta_estimate'] = delta_est
        cur_data['validation_diff_svals'] = validation_diff_svals
        cur_data['true_diff_svals'] = true_diff_svals
        cur_data['time_diff'] = time_diff
        cur_data.update(result)

        out_df = out_df.append(cur_data, ignore_index=True)
        if out_df is not None: 
            out_df.to_csv(out_path)
    return out_df


####################
## post-processing
##################
def parse_str_array(arr): 
    s = arr.replace('[', '')
    s = s.replace(']', '')
    s = s.replace('\n', '')

    return [np.float(x) for x in s.strip().split(' ') if len(x) > 0]

def post_process_df(df): 
    if isinstance(df['validation_diff_svals'].iloc[0], str): 
        df['validation_diff_svals'] = df['validation_diff_svals'].map(lambda x: parse_str_array(x))
    if isinstance(df['true_diff_svals'].iloc[0], str): 
        df['true_diff_svals'] = df['true_diff_svals'].map(lambda x: parse_str_array(x))
    df['val_diff_max_sv'] = df['validation_diff_svals'].map(lambda x: max(x))
    df['true_diff_max_sv'] = df['true_diff_svals'].map(lambda x: max(x))
    df['val_diff_max_sv_normalized'] = np.divide(df['val_diff_max_sv'], df['l2_diff_val_gt'])
    df['true_diff_max_sv_normalized'] = np.divide(df['true_diff_max_sv'], df['l2_diff_val_gt'])

    df['total_true_loss_twenty_svals'] = df['true_diff_svals'].map(lambda x: sum(x))
    df['total_true_loss_normalized'] = np.divide(df['total_true_loss_twenty_svals'], df['l2_diff_val_gt'])

    df['total_val_loss_twenty_svals'] = df['true_diff_svals'].map(lambda x: sum(x))
    df['total_val_loss_normalized'] = np.divide(df['total_val_loss_twenty_svals'], df['l2_diff_val_gt'])
    return df

def lineplot_results(df, num_eigs_included = 10, savepath=None): 
    sns.lineplot(x = 'm', y ='true_diff_max_sv_normalized', data=df, hue='occlusion_level', style='delta_scaling', 
                palette="crest")
    plt.legend(bbox_to_anchor=(1, 1))
    plt.title('Performance on Occlusion Test, No $A^{(s)}$ Components, Optimizing for {} SVals'.format(num_eigs_included))
    plt.xlabel('Num samples')
    plt.ylabel('Ratio of $\| \hat P - P \|$ to $\| A^{(s)} - P \|$')
    plt.tight_layout()
    if savepath is not None: 
        plt.savefig(savepath, dpi=500.0)

#################
### main 
##################
def main(): 
    now = datetime.now()
    out_path = 'logs/job_{}.csv'.format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    print('Writing to {}'.format(out_path))
    out_df = run_tests_one_ground_truth(out_path=out_path)
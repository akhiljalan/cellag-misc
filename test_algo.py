import unittest   # The test framework
import graph_learning_utils as gl
import numpy as np
import scipy 

class Test_BasicAlgoTests(unittest.TestCase): 
    def test_finds_validation_matrix(self):
        n = 20
        ground_truth_asymm = np.random.uniform(0, 0.5, (n, n))
        ground_truth_mat = ground_truth_asymm + ground_truth_asymm.T

        m = 10
        sample_mats = [gl.gen_sample_mat(ground_truth_mat) for _ in range(m)]

        sample_mats = [ground_truth_mat] + sample_mats
        eta_init = gl.generate_random_eta(len(sample_mats))

        result = gl.run_scipy_minimize(
            eta_init, sample_mats, ground_truth_mat, delta=0.0)
        
        assert(np.isclose(result['fun'], 0.0, atol=1e-6),
                'Objective should be 0.0 but is {}'.format(result['fun']))
        assert(np.isclose(result['x'][0], 1.0, atol=1e-6), 
               'First entry of eta should be 1.0 but is {}'.format(result['x'][0]))
        assert(np.allclose(result['x'][1:], np.zeros(len(sample_mats) - 1), atol=1e-6), 
               'Other entries of eta should be 0.0 but are {}'.format(result['x'][1:]))
        
    def test_finds_half_split(self): 
        eigenbasis_matrix = scipy.stats.ortho_group.rvs(20)
        mat1 = eigenbasis_matrix @ np.diag([1.0] + [0.1 for _ in range(19)]) @ eigenbasis_matrix.T
        mat2 = eigenbasis_matrix @ np.diag([0.1, 1.0] + [0.1 for _ in range(18)]) @ eigenbasis_matrix.T
        mat = 0.5 * mat1 + 0.5 * mat2
        sample_mats = [mat1, mat2]
        eta_init = gl.generate_random_eta(len(sample_mats))

        result = gl.run_scipy_minimize(eta_init, sample_mats, mat, delta=0.0)

        assert(np.isclose(result['fun'], 0.0, atol=1e-6),
                'Objective should be 0.0 but is {}'.format(result['fun']))
        assert(np.allclose(result['x'], 0.5 * np.ones_like(result['x']), atol=1e-6), 
               'All entries of eta should be 0.5 but are {}'.format(result['x']))
        
    def test_finds_half_split_imbalanced_no_simplex_constraint(self): 
        eigenbasis_matrix = scipy.stats.ortho_group.rvs(20)
        mat1 = eigenbasis_matrix @ np.diag([1.0] +
                                        [0.1 for _ in range(19)]) @ eigenbasis_matrix.T
        mat2 = eigenbasis_matrix @ np.diag([0.1, 1.0] +
                                        [0.1 for _ in range(18)]) @ eigenbasis_matrix.T
        mat = 0.5 * mat1 + 0.5 * mat2


        sample_mats = [0.4 * mat1, 0.2 * mat1, mat2]
        eta_init = gl.generate_random_eta(len(sample_mats) + 1)
        objective = lambda eta: gl.objective_with_params_scale_x0(eta, mat, sample_mats, 0.00)

        result = scipy.optimize.minimize(
            objective,
            eta_init,
            method='SLSQP',
            jac=None,
            bounds=[(0.0, 1)] + [(0, 1) for _ in range(len(sample_mats))],
            options={
                'maxiter': 10000,
                'disp': False
            }
        )
        coeff_scale = result['x'][1] * 0.4 + result['x'][2] * 0.2 - result['x'][3] * 1.0
        assert np.isclose(coeff_scale, 0.0, atol=1e-5), 'Scaled coeffs should sum to 0.0 but are {}'.format(coeff_scale)
        assert np.isclose(result['fun'], 0.0, atol=1e-5), 'Objective should be 0.0 but is {}'.format(result['fun'])
        svd_values = scipy.linalg.svdvals(result['x'][0] * mat 
                                            - gl.matrix_lin_combo_pos_sign(result['x'][1:], sample_mats))
        # print('without simplex constraint', svd_values)
        assert np.allclose(svd_values, np.zeros_like(svd_values), atol=1e-5), 'SVD values of diff should all be near 0.0 but are {}'.format(svd_values)
        
    def test_finds_half_split_imbalanced_with_simplex_constraint(self):
        eigenbasis_matrix = scipy.stats.ortho_group.rvs(20)
        mat1 = eigenbasis_matrix @ np.diag([1.0] +
                                           [0.1 for _ in range(19)]) @ eigenbasis_matrix.T
        mat2 = eigenbasis_matrix @ np.diag([0.1, 1.0] +
                                           [0.1 for _ in range(18)]) @ eigenbasis_matrix.T
        mat = 0.5 * mat1 + 0.5 * mat2

        sample_mats = [0.4 * mat1, 0.2 * mat1, mat2]
        eta_init = gl.generate_random_eta(len(sample_mats) + 1)
        objective = lambda eta: gl.objective_with_params_scale_x0(eta, mat, sample_mats, 0.00)

        result = scipy.optimize.minimize(
            objective,
            eta_init,
            method='SLSQP',
            jac=None,
            constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
            bounds=[(0, 1) for _ in range(len(sample_mats) + 1)],
            options={
                'maxiter': 10000,
                'disp': False
            }
        )
        coeff_scale = result['x'][1] * 0.4 + result['x'][2] * 0.2 - result['x'][3] * 1.0
        tol = 1e-5
        assert np.isclose(coeff_scale, 0.0, atol=tol), 'Scaled coeffs should sum to 0.0 but are {}'.format(coeff_scale)
        assert np.isclose(result['fun'], 0.0, atol=tol), 'Objective should be 0.0 but is {}'.format(result['fun'])
        svd_values = scipy.linalg.svdvals(result['x'][0] * mat 
                                            - gl.matrix_lin_combo_pos_sign(result['x'][1:], sample_mats))
        # print('with simplex constraint', svd_values)
        assert np.allclose(svd_values, np.zeros_like(svd_values), atol=tol), 'SVD values of diff should all be near 0.0 but are {}'.format(svd_values)

if __name__ == '__main__':
    unittest.main()

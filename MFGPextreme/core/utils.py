import numpy as np
import scipy
from KDEpy import FFTKDE
from scipy.interpolate import interp1d
import os


# custom_KDE and set_worker_env are based on gpsearch
def custom_KDE(data, weights=None, bw=None):
    data = data.flatten()
    if bw is None:
        try:
            sc = scipy.stats.gaussian_kde(data, weights=weights)
            bw = np.sqrt(sc.covariance).flatten()
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0
    return FFTKDE(bw=bw).fit(data, weights)


def set_worker_env(n_threads=1):
    """Prevents over-subscription in joblib."""
    MAX_NUM_THREADS_VARS = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for var in MAX_NUM_THREADS_VARS:
        os.environ[var] = str(n_threads)


def compute_tf_cost(X_list, c_ratio, full_return=False):
    cost_list = np.zeros(len(X_list))
    if not full_return:
        for ii, X in enumerate(X_list):
            num_h_X = np.count_nonzero(X[:,-1]==1)
            num_l_X = np.count_nonzero(X[:,-1]==0)
            cost_list[ii] = num_h_X +  1 / c_ratio * num_l_X
        return cost_list
    else:
        cost_list_h = np.zeros(len(X_list))
        cost_list_l = np.zeros(len(X_list))
        for ii, X in enumerate(X_list):
            num_h_X = np.count_nonzero(X[:,-1]==1)
            num_l_X = np.count_nonzero(X[:,-1]==0)
            cost_l = 1 / c_ratio * num_l_X
            cost_list[ii] = num_h_X + cost_l
            cost_list_h[ii] = num_h_X
            cost_list_l[ii] = cost_l
        return cost_list, cost_list_h, cost_list_l

def generate_plotdata(result_runs, c_ratio, fixed_quato=False):
    '''
    Generate data for plot 
    
    Parameters
    ----------
    result_runs: list of result
        result: [data_list, error_list]
    c_ratio: float
        high to low cost ratio
    fixed_quato: bool    
    
    Return
    ----------
    cost_list: array
    error_list_runs: [error_list_1, error_list_2....]    
    '''
    error_list_runs = [result[1] for result in result_runs]
    if fixed_quato:
        cost_list = compute_tf_cost(result_runs[0][0], c_ratio)
    else:
        # compute the range for cost_list
        cost_list_runs = [compute_tf_cost(result[0], c_ratio) 
                                            for result in result_runs]
        end1 =  np.max([cost_list[0] for cost_list in cost_list_runs])
        end2 = np.min([cost_list[-1] for cost_list in cost_list_runs])
        cost_list = np.arange(end1, end2, 0.2)
        # re-compute the error
        error_list_runs = [interp1d(i,j)(cost_list) 
                                for i,j in zip(cost_list_runs, error_list_runs)]
    return cost_list, np.asarray(error_list_runs, float)
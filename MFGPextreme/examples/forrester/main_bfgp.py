import sys
sys.path.append("../../")
import warnings
import os

import numpy as np
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel as C
from joblib import Parallel, delayed
from core import (GaussianProcessRegressor, RBF_, TFkernel, AcqLW,
                  OptimalDesignTF, GaussianInputs, custom_KDE, log_pdf,
                  set_worker_env)
from examples.forrester.forrester import f_h, f_l

def main():
    
    
    n_trails = 100  # number of trails
    n_seq = 60      # maximum number of sequential samples       
    n_cost = 12     # maximum cost
    c_ratio = 5     # h/l cost ratio

    # def input
    dim = 1
    mean = np.array([0.5])
    cov = np.array([[0.1**2]])
    domain = np.array([[0,1]])
    inputs = GaussianInputs(mean, cov, domain, dim)

    # The true pdf
    smpl = np.loadtxt('map_samples.txt') 
    pts = smpl[:,0:-1]
    yy = smpl[:,-1]
    weights = inputs.pdf(pts)
    pt = custom_KDE(yy, weights=weights)

    # def kernel 
    kernel_l = RBF_(C(1e1, (1e-1, 1e2)), RBF((1,), (1e-1, 1e1)))
    kernel_d = RBF_(C(1e1, (1e-1, 1e3)), RBF((1,), (1e-1, 1e2)))
    likelihood_l = WhiteKernel(1e-10, 'fixed')
    likelihood_h = WhiteKernel(1e-10, 'fixed')
    exp_rho = C(np.exp(2), 'fixed')
    tfkernel = TFkernel([kernel_l, kernel_d, likelihood_l, 
                                                     likelihood_h, exp_rho])

    # def initial samples
    n_init_h, n_init_l = 2, 10


    def wrapper_bm(trail):
        warnings.filterwarnings("ignore")
        tfgp = GaussianProcessRegressor(tfkernel, n_restarts_optimizer=6)
        acq = AcqLW(inputs)
        np.random.seed(trail)
        opt = OptimalDesignTF(f_h, f_l, inputs)
        opt.init_sampling(n_init_h, n_init_l)
        models = opt.seq_sampling_opt(n_seq, n_cost, c_ratio, acq, tfgp,
                                                n_starters=6, n_jobs=1)
        errors = log_pdf(models, pt, pts, weights)
        datas = [model.X_train_ for model in models]
        #print(trail)
        return datas, errors

    set_worker_env()
    results = Parallel(n_jobs=20)(delayed(wrapper_bm)(j)
                                     for j in range(n_trails))
    os.makedirs('data', exist_ok=True)
    np.save('data/opt', results)
    

if __name__=='__main__':
    main()

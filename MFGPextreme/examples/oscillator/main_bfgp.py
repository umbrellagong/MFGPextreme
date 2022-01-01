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
from examples.oscillator.oscillator import f_l, f_h

def main(strategy, n_seq, n_ratio=5, n_cost=80, c_ratio=5):
    '''
    strategy must be one of 
            (1) 'fix' with arguments n_seq, n_ratio
            (2) 'opt' with arguments n_seq, n_cost, c_ratio
    '''  
    # def number of trails
    n_trails = 100

    # def input
    dim = 2
    mean = np.zeros(2)
    cov = np.eye(2)
    domain = np.array([[-6,6]]*dim)
    inputs = GaussianInputs(mean, cov, domain, dim)

    # The true pdf 
    smpl = np.loadtxt('map_samples.txt') 
    pts = smpl[:,0:-1]
    yy = smpl[:,-1]
    weights = inputs.pdf(pts)
    pt = custom_KDE(yy, weights=weights)

    # def kernel 
    kernel_l = RBF_(C(1e1, (1e-1,1e2)), RBF((1,1), (1e-1, 2*1e1)))
    kernel_d = RBF_(C(1e0, (1e-1,1e0)), RBF((1,1), (1e-1, 2*1e1)))
    likelihood_l = WhiteKernel(1e-10, 'fixed')
    likelihood_h = WhiteKernel(1e-10, 'fixed')
    exp_rho = C(np.exp(1), (np.exp(0), np.exp(2)))
    tfkernel = TFkernel([kernel_l, kernel_d, likelihood_l, 
                                                     likelihood_h, exp_rho])

    # def initial samples
    n_init_h, n_init_l = 4, 20
    
    # generate results for each random seed
    def wrapper_bm(trail):
        warnings.filterwarnings("ignore")
        tfgp = GaussianProcessRegressor(tfkernel, n_restarts_optimizer=6)
        acq = AcqLW(inputs)
        np.random.seed(trail)
        opt = OptimalDesignTF(f_h, f_l, inputs)
        opt.init_sampling(n_init_h, n_init_l)

        if strategy=='fix':
            models = opt.seq_sampling_fixed(n_seq, n_ratio, acq, tfgp, 
                                                    n_starters=6, n_jobs=1)
        elif strategy=='opt':
            models = opt.seq_sampling_opt(n_seq, n_cost, c_ratio, acq, tfgp, 
                                                    n_starters=6, n_jobs=1)
        errors = log_pdf(models, pt, pts, weights)
        datas = [model.X_train_ for model in models]        
        
        return datas, errors 
    
    set_worker_env()
    results = Parallel(n_jobs=20)(delayed(wrapper_bm)(j)
                                     for j in range(n_trails))
    
    os.makedirs('data/amp01', exist_ok=True)
    FILENAME = strategy + '_' + str(n_ratio) +'_amp01'
    np.save('data/amp01/' + FILENAME, results)


if __name__=='__main__':
    #opt
    main('opt', 400)
    #fixed
    main('fix', 60, 1)
    main('fix', 52, 2)
    main('fix', 36, 5)
    main('fix', 24, 10)
    main('fix', 18, 15)
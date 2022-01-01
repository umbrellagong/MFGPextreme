import sys
sys.path.append("../../")
import os

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from joblib import Parallel, delayed
from core import (GaussianProcessRegressor, RBF_, AcqLW, OptimalDesign, 
                  GaussianInputs, custom_KDE, log_pdf, set_worker_env)
from examples.oscillator.oscillator import f_h


def main():
    
    # def trails
    n_trails = 100

    # build the input
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

    # build the kernel
    kernel = RBF_(C(1e1, (1e-1,1e2)), RBF((1,1), (1e-1, 1e2)))

    # sequential details
    n_init=8
    n_seq=72

    
    def wrapper_bm(trail):
        np.random.seed(trail)
        sgp = GaussianProcessRegressor(kernel, n_restarts_optimizer=6)
        acq = AcqLW(inputs)
        np.random.seed(trail)
        opt = OptimalDesign(f_h, inputs)
        opt.init_sampling(n_init)
        models = opt.seq_sampling(n_seq, acq, sgp, n_starters=6, n_jobs=1)
        error = log_pdf(models, pt, pts, weights, tf=False)
        return models, error

    set_worker_env()
    results = Parallel(n_jobs=20)(delayed(wrapper_bm)(j)
                                     for j in range(n_trails))
    os.makedirs('data', exist_ok=True)
    np.save('data/sgp', results)

if __name__=='__main__':
    main()
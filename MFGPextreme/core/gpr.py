import numpy as np
import sklearn.gaussian_process as skgp
from scipy.linalg import cho_solve
import scipy.optimize
import warnings
from scipy.linalg import cholesky, cho_solve
from sklearn.utils.validation import check_array
from sklearn.utils.optimize import _check_optimize_result
from sklearn.preprocessing._data import _handle_zeros_in_scale
import time


class GaussianProcessRegressor(skgp.GaussianProcessRegressor):
    '''Customized gpr from sklearn

    Add gradient computation, fixed hyperparameter fit.
    '''


    def post_cov(self, x):
        '''Compute gradient of the posterior covariance.

        Usefull in computing the derivative of acquisition. 
        '''
        K_trans = self.kernel_(x, self.X_train_)
        V = cho_solve((self.L_, True), K_trans.T)
        term1 = self.kernel_.gradient_x(x, x)
        term2 = 2 * V.T.dot(self.kernel_.gradient_x(x, self.X_train_))
        gradient = term1 - term2
        cov = self.kernel_(x,x) - K_trans.dot(V)
        return  cov, gradient

    def fit_keep(self, X, y):
        '''Same GPR with different dataset.

        '''
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            self._y_train_std = _handle_zeros_in_scale(np.std(y, axis=0), 
                                                       copy=False)
            # Remove mean and make unit variance
            y = (y - self._y_train_mean) / self._y_train_std
        else:
            self._y_train_mean = np.zeros(1)
            self._y_train_std = 1 
        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return self

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        '''A light optimization. 
        '''
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options = {'maxiter':100}
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, 
                                                 bounds=bounds)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}.")

        return theta_opt, func_min


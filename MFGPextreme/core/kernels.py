import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel, Product, Sum
from scipy import linalg
from scipy.spatial import distance
import copy


class RBF_(Product):   
    '''
    An augmented RBF kernel including kernel integrations and derivative.  

    The Product and its father kerneloperator defines K, K_gradient, theta and
    bounds.
    '''

    def gradient_x(self, x, X): 
        '''Compute the gradient of k(X, x) to x 
        
        Parameter
        ----------
        x : array (1,d)
        X : array (n,d)

        Return
        ---------
        gradients: array (n,d)
        '''

        Lambda = np.atleast_1d(self.get_params()['k2__length_scale'])**2
        return np.dot(X - x, np.diag(1/Lambda)) * self.__call__(X, x)
        
    def intKNorm(self, X, mean, cov):
        '''Compute \int K(X, x')N(x'; mean, cov) dx'
        
        Parameter
        ----------
        X : array (n,d)

        Return
        ---------
        res: array (n,)
        '''

        mean = np.atleast_2d(mean) 
        Lambda = np.atleast_1d(self.get_params()['k2__length_scale'])**2
        kappa = linalg.det(np.dot(cov, np.diag(1/Lambda)) 
                                  + np.eye(mean.shape[1]))**(-1/2)
        # can also be solved by sum(a * b)
        kappa *= np.exp(-distance.cdist(X, mean, 'mahalanobis', 
                           VI=linalg.inv(cov + np.diag(Lambda)))**2 / 2)
        kappa *= self.get_params()['k1__constant_value']
        return kappa.reshape(-1) 

    def intKKNorm(self, X1, X2, mean, cov):
        '''Compute \int K(X1, x')K(x',X2)N(x'; mean, cov) dx'
        
        Parameter
        ----------
        X1 : array (n1,d)
        X2 : array (n2,d)

        Return
        ---------
        res: array (n1, n2)
        '''

        mean = np.atleast_2d(mean) # 2d
        Lambda = np.atleast_1d(self.get_params()['k2__length_scale'])**2
        var = self.get_params()['k1__constant_value']
        const = linalg.det(2*np.dot(cov, np.diag(1/Lambda)) 
                           + np.eye(mean.shape[1]))**(-1/2)
        k1 = self(X1/np.sqrt(2), X2/np.sqrt(2))
        k2 = np.exp(-distance.cdist(X1/2 - mean, -X2/2, 'mahalanobis', 
                               VI=linalg.inv(cov + np.diag(Lambda)/2))**2 / 2)
        kappa = var * const * k1 * k2
        return kappa

    
    def intK1K2Norm(self, X1, X2, kernel2, mean, cov): 
        '''Compute \int K1(X1, x')K2(x',X2)N(x'; mean, cov) dx'
        
        Parameter
        ----------
        X1 : array (n1,d)
        X2 : array (n2,d)

        Return
        ---------
        res: array (n1, n2)
        '''
        mean = np.atleast_2d(mean) # 2d
        scale1 = np.atleast_1d(self.get_params()['k2__length_scale'])**2
        scale2 = np.atleast_1d(kernel2.get_params()['k2__length_scale'])**2
        var1 = self.get_params()['k1__constant_value']
        var2 = kernel2.get_params()['k1__constant_value']

        C = scale1 * scale2 / (scale1 + scale2) #(d,)
        m1 = scale2 / (scale1 + scale2) #(d,)
        m2 = scale1 / (scale1 + scale2) #(d,)

        const = linalg.det(np.dot(cov, np.diag(1/C)) 
                           + np.eye(mean.shape[1]))**(-1/2)
        k1 = np.exp(-distance.cdist(X1, X2, 'mahalanobis', 
                                    VI=np.diag(1 / (scale1+scale2))) **2 / 2)
        k2 = np.exp(-distance.cdist(X1*m1 - mean, -X2*m2, 'mahalanobis', 
                                    VI=linalg.inv(cov + np.diag(C)))**2 / 2)
        kappa = var1 * var2 * const * k1 * k2 
        return kappa


    def dintKKNorm_dx(self, x, X, mean, cov):
        return self.dintK1K2Norm_dx(x, X, self, mean, cov)
        
    

    def dintK1K2Norm_dx(self, x, X, kernel2, mean, cov): 
        '''Compute d (\int K1(x, x')K2(x', X)N(x'; mean, cov) dx') dx
        
        A corestone for multi-fidelity kernel.

        Parameter
        ----------
        x : array (1,d)
        X : array (n,d)

        Return
        ---------
        res: array (n, d)
        '''
        integral = self.intK1K2Norm(x, X, kernel2, mean, cov) # (1,n) matrix
        scale1 = np.atleast_1d(self.get_params()['k2__length_scale'])**2
        scale2 = np.atleast_1d(kernel2.get_params()['k2__length_scale'])**2

        C = scale1 * scale2 / (scale1 + scale2) #(d,)
        m1 = scale2 / (scale1 + scale2) #(d,)
        m2 = scale1 / (scale1 + scale2) #(d,)

        gradient = (np.dot(X-x, np.diag(1/(scale1+scale2))) + 
            np.dot(mean-(x*m1 + X*m2), linalg.inv(cov + np.diag(C))).dot(
                                      np.diag(scale2 / (scale1 + scale2)))
                          )*integral.T
        return gradient



class TFkernel(CompoundKernel):
    '''
    A bi-fidelity kernel.
    
    Kernel for autogressive model where f_h = rho * f_l + d. The extention to
    general multi-fidelity situation is coming soon. 

    parameter
    ---------
    A list of kernels including 
        rbf_l, rbf_d: instances of RBF_
        ll_l, ll_d: instances of Whitekernel
        exp_rho: instance of Constantkernel
            e^rho
    '''
    def __init__(self, kernels):
        self.rbf_l = kernels[0]
        self.rbf_d = kernels[1]
        self.ll_l = kernels[2]
        self.ll_h = kernels[3]
        self.exp_rho = kernels[4]
        # To use the theta properties to change and read hyperparameters
        self.kernels = kernels

    # The bounds in CompoundKernel are wrong. Need re-write.
    @property
    def bounds(self):
        bounds_ = np.empty((0,2))
        for kernel in self.kernels:
            if kernel.bounds.size != 0:     # unfixed only
                bounds_ = np.vstack((bounds_, kernel.bounds))
        return bounds_

    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Compute the kernel value and gradient w.r.t. hyperparameters.

        parameter
        ----------
        X: array (n1, d+1)
            augmented with the fidelity: 1 for high-fidelity and 0 for low.
        Y: array (n2, d+1)
            augmented with the fidelity: 1 for high-fidelity and 0 for low.

        return 
        ---------
        K: array (n1, n2)
            kernel value
        grad: array (n1, n2, d)
            gradient of kernel w.r.t. hyperparameters

        '''
        num_h_X = np.count_nonzero(X[:,-1]==1)    # Number of hf samples
        rho = np.log(self.exp_rho.constant_value) # The true rho
        X = np.copy(X)[:,:-1]                     # protect the original X
        if Y is None:                             
            if eval_gradient:  
                K_l, dK_rbf_l = self.rbf_l(X, eval_gradient=True)
                dK_rho = np.copy(K_l)
                K_d, dK_rbf_d_ = self.rbf_d(X[:num_h_X], eval_gradient=True)
                
                K_llh, dK_llh_= self.ll_h(X[:num_h_X], eval_gradient=True)
                K_lll, dK_lll_= self.ll_l(X[num_h_X:], eval_gradient=True)

                K = self.compute_K(K_l, K_d, rho, num_h_X, K_lll, K_llh)

                # always have the rbf_l and rbf_d unfixed
                dK_rbf_l[:num_h_X, :num_h_X] *= rho**2
                dK_rbf_l[:num_h_X, num_h_X:] *= rho
                dK_rbf_l[num_h_X:, :num_h_X] *= rho
                dK_rbf_d = np.zeros((X.shape[0], X.shape[0], 
                                     len(self.rbf_d.theta)))
                dK_rbf_d[:num_h_X, :num_h_X, :] = dK_rbf_d_

                # ll_l, ll_h and rho could be fixed and their derivative would
                # be empty.
                if self.ll_h.hyperparameters[0].fixed:
                    dK_llh = np.empty((X.shape[0], X.shape[0], 0))
                else:
                    dK_llh = np.zeros((X.shape[0], X.shape[0], 1))
                    dK_llh[:num_h_X, :num_h_X, 0] = dK_llh_

                if self.ll_l.hyperparameters[0].fixed:
                    dK_lll = np.empty((X.shape[0], X.shape[0], 0))
                else:
                    dK_lll = np.zeros((X.shape[0], X.shape[0], 1))
                    dK_lll[num_h_X:, num_h_X:, 0] = dK_lll_
                
                if self.exp_rho.hyperparameters[0].fixed:
                    dK_rho = np.empty((X.shape[0], X.shape[0], 0))
                else: 
                    dK_rho[:num_h_X, :num_h_X] *= 2*rho
                    dK_rho[num_h_X:, num_h_X:] = 0
                    dK_rho = dK_rho[:,:,np.newaxis]
                # assemble the derivatives
                dK_hyp = np.concatenate([dK_rbf_l, dK_rbf_d, 
                                         dK_lll, dK_llh, dK_rho], axis=2)
                return K, dK_hyp

            else:
                # No derivative needed
                K_l = self.rbf_l(X)             
                K_d = self.rbf_d(X[:num_h_X])   
                K_llh = self.ll_h(X[:num_h_X])  
                K_lll = self.ll_l(X[num_h_X:])
                K = self.compute_K(K_l, K_d, rho, num_h_X, K_lll, K_llh)
                return K
        else:
            if eval_gradient:
                raise('wrong!')
            num_h_Y = np.count_nonzero(Y[:,-1]==1)
            Y = np.copy(Y)[:,:-1]

            K_l = self.rbf_l(X, Y)
            K_d = self.rbf_d(X[:num_h_X], Y[:num_h_Y])
            K = self.compute_K(K_l, K_d, rho, num_h_X, num_h_Y=num_h_Y)
            return K

    def diag(self, X): 
        num_h_X = np.count_nonzero(X[:,-1]==1)
        rho = np.log(self.exp_rho.constant_value)
        X = np.copy(X)[:,:-1]
        diag_vec = self.rbf_l.diag(X)
        diag_vec[:num_h_X] *= rho**2
        diag_vec[:num_h_X] += self.rbf_d.diag(X[:num_h_X])
        diag_vec[:num_h_X] += self.ll_h.diag(X[:num_h_X])
        diag_vec[num_h_X:] += self.ll_l.diag(X[num_h_X:])
        return diag_vec


    @staticmethod
    def compute_K(K_l, K_d, rho, num_h_X, K_lll=0, K_llh=0, num_h_Y=None):
        if num_h_Y==None:
            num_h_Y = num_h_X
        
        K = K_l
        K[:num_h_X, :num_h_Y] *= rho**2                  
        K[:num_h_X, :num_h_Y] += K_d
        K[:num_h_X, :num_h_Y] += K_llh
        K[:num_h_X, num_h_Y:] *= rho
        K[num_h_X:, :num_h_Y] *= rho
        K[num_h_X:, num_h_Y:] += K_lll
        return K

    @property
    def theta(self):   # unfixed hyper-parameters
        return np.hstack([kernel.theta for kernel in self.kernels])

    @theta.setter
    def theta(self, theta):
        current_dims = 0
        for kernel in self.kernels:
            k_dims = kernel.n_dims
            kernel.theta = theta[current_dims: current_dims + k_dims]
            current_dims += k_dims

    def _check_bounds_params(self):
        pass


    def intKNorm(self, X, mean, cov): # X is augmented data. 
        num_h = np.count_nonzero(X[:,-1]==1)
        rho = np.log(self.exp_rho.constant_value)  
        X = np.copy(X)[:,:-1]
        kappa_l = self.rbf_l.intKNorm(X, mean, cov) 
        kappa_d = self.rbf_d.intKNorm(X[:num_h], mean, cov)
        kappa = kappa_l * rho
        kappa[:num_h] *= rho
        kappa[:num_h] += kappa_d
        return kappa
    
    def intKKNorm(self, X1, X2, mean, cov):
        '''Compute \int cov(Y(X1),f_h(x'))cov(f_h(x'),Y(X2))N(x'; mean, cov)dx'.
        
        Based on int K1K2 norm.

        Parameter
        ----------
        X1 : array (n1,d+1)
        X2 : array (n1,d+1)

        Return
        ---------
        res: array (n1, n2)
        '''
        num_h1 = np.count_nonzero(X1[:,-1]==1) 
        num_h2 = np.count_nonzero(X2[:,-1]==1)
        X1 = np.copy(X1)[:,:-1]
        X2 = np.copy(X2)[:,:-1]
        rho = np.log(self.exp_rho.constant_value)
        
        kappa = rho**2 * self.rbf_l.intKKNorm(X1, X2, mean, cov)
        kappa[:num_h1, :num_h2] *= rho**2
        kappa[:num_h1, :num_h2] += self.rbf_d.intKKNorm(X1[:num_h1], 
                                                        X2[:num_h2], 
                                                        mean, cov)
        kappa[:num_h1, :num_h2] += rho**2 *self.rbf_l.intK1K2Norm(X1[:num_h1], 
                                                                    X2[:num_h2],
                                                                    self.rbf_d,
                                                                    mean, cov)
        kappa[:num_h1, :num_h2] += rho**2 *self.rbf_d.intK1K2Norm(X1[:num_h1], 
                                                                    X2[:num_h2],
                                                                    self.rbf_l,
                                                                    mean, cov)
        
        kappa[:num_h1, num_h2:] *= rho
        kappa[:num_h1, num_h2:] += rho * self.rbf_d.intK1K2Norm(X1[:num_h1], 
                                                                X2[num_h2:],
                                                                self.rbf_l,
                                                                mean, cov)
        kappa[num_h1:, :num_h2] *= rho
        kappa[num_h1:, :num_h2] += rho * self.rbf_l.intK1K2Norm(X1[num_h1:], 
                                                                X2[:num_h2],
                                                                self.rbf_d,
                                                                mean, cov)
                                                    
        return kappa 

    def gradient_x(self, x, X):
        '''compute d cov(f_i(pos), Y(X)) d pos
        '''
        num_h_X = np.count_nonzero(X[:,-1]==1) 
        rho = np.log(self.exp_rho.constant_value)
        X = np.copy(X)[:,:-1]
        num_h_x = np.count_nonzero(x[:,-1]==1)
        x = np.copy(x)[:,:-1]
        
        gradient = self.rbf_l.gradient_x(x,X)  
        if num_h_x==1:   
            gradient[:num_h_X] *= rho**2
            gradient[:num_h_X] += self.rbf_d.gradient_x(x, X[:num_h_X])
            gradient[num_h_X:] *= rho
        else:
            gradient[:num_h_X] *= rho
        return gradient

    def dintKKNorm_dx(self, x, X, mean, cov):
        '''Compute d \int cov(f_i(pos), x')cov(x', Y)N(x';mean,cov)dx' d pos
        '''
        num_h_X = np.count_nonzero(X[:,-1]==1)
        rho = np.log(self.exp_rho.constant_value)
        X = np.copy(X)[:,:-1]
        num_h_x = np.count_nonzero(x[:,-1]==1)
        x = np.copy(x)[:,:-1]

        gradient = rho**2 * self.rbf_l.dintK1K2Norm_dx(x, X, self.rbf_l, 
                                                              mean, cov)
        if num_h_x==1:   # x is a high-fidelity sample
            gradient[:num_h_X] *= rho**2
            gradient[:num_h_X] += self.rbf_d.dintK1K2Norm_dx(x, X[:num_h_X],
                                                             self.rbf_d,
                                                             mean, 
                                                             cov)
            gradient[:num_h_X] += rho**2 * self.rbf_l.dintK1K2Norm_dx(x, 
                                                                  X[:num_h_X],
                                                                  self.rbf_d,
                                                                  mean,
                                                                  cov)
            gradient[:num_h_X] += rho**2 * self.rbf_d.dintK1K2Norm_dx(x, 
                                                                  X[:num_h_X],
                                                                  self.rbf_l,
                                                                  mean,
                                                                  cov)
            gradient[num_h_X:] *= rho
            gradient[num_h_X:] += rho* self.rbf_d.dintK1K2Norm_dx(x, 
                                                                  X[num_h_X:],
                                                                  self.rbf_l,
                                                                  mean,
                                                                  cov)
        else:
            gradient[:num_h_X] *= rho
            gradient[:num_h_X] += rho* self.rbf_l.dintK1K2Norm_dx(x, 
                                                                  X[:num_h_X],
                                                                  self.rbf_d,
                                                                  mean,
                                                                  cov)
        return gradient


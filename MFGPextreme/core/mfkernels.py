import numpy as np
from sklearn.gaussian_process.kernels import CompoundKernel, Product, Sum
from scipy import linalg
from scipy.spatial import distance
import pdb

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



class MFkernel(CompoundKernel):
    '''
    A milti-fidelity kernel.
    '''
    def __init__(self, kernels):
        self.n_level = int(len(kernels) + 1) // 2
        self.rbfs = kernels[:self.n_level] # rbf kernels
        self.exp_rhos = kernels[self.n_level:]  # exp(rhos) 
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

        rhos = [np.log(exp_rho.constant_value) for exp_rho in self.exp_rhos]
        num_h_X = [0] + [np.count_nonzero(X[:,-1]== i) for i in range(self.n_level)]   
        num_h_X = np.cumsum(num_h_X)
        
        if eval_gradient:
            X = np.copy(X)[:,:-1]
            K = np.zeros((X.shape[0], X.shape[0]))
            dK_drbfs = [np.zeros((X.shape[0], X.shape[0], self.rbfs[i].n_dims)) 
                        for i in range(self.n_level)]
            dK_drhos = [np.zeros((X.shape[0], X.shape[0], self.exp_rhos[i].n_dims)) 
                        for i in range(self.n_level-1)]

            for k in range(self.n_level):
                K_k = np.zeros((X.shape[0], X.shape[0]))
                K_k[num_h_X[k]:, num_h_X[k]:], dK_drbfs[k][num_h_X[k]:, num_h_X[k]:] = \
                    self.rbfs[k](X[num_h_X[k]:,:], eval_gradient=True)

                for i in range(k, self.n_level):
                    for j in range(k, self.n_level):
                        scale1 = np.prod(rhos[k:i]) 
                        scale2 = np.prod(rhos[k:j]) 
                        K_k[num_h_X[i]: num_h_X[i+1],
                                num_h_X[j] : num_h_X[j+1]] *= scale1 * scale2
                        dK_drbfs[k][num_h_X[i]: num_h_X[i+1],
                                num_h_X[j] : num_h_X[j+1]] *= scale1 * scale2
                        # compute the gradient of rho
                        for l in range(k, i):
                            if self.exp_rhos[l].n_dims == 1:
                                if rhos[l] == 0:
                                    raise ValueError("The hyperparameter rho"
                                        " should be positive.")
                                dK_drhos[l][num_h_X[i]: num_h_X[i+1],
                                        num_h_X[j] : num_h_X[j+1], 0] += \
                                    K_k[num_h_X[i]: num_h_X[i+1], 
                                        num_h_X[j] : num_h_X[j+1]] / rhos[l]
                        for l in range(k, j):
                            if self.exp_rhos[l].n_dims == 1:
                                if rhos[l] == 0:
                                    raise ValueError("The hyperparameter rho"
                                        " should be positive.")
                                dK_drhos[l][num_h_X[i]: num_h_X[i+1],
                                        num_h_X[j] : num_h_X[j+1], 0] += \
                                    K_k[num_h_X[i]: num_h_X[i+1], 
                                        num_h_X[j] : num_h_X[j+1]] / rhos[l]    
                K += K_k
            #pdb.set_trace()
            dK_dtheta = np.concatenate((np.concatenate(dK_drbfs, axis=2), 
                            np.concatenate(dK_drhos, axis=2)),
                            axis=2)
            return K, dK_dtheta

        if Y is None:
            Y = X
        X = np.copy(X)[:,:-1]
        num_h_Y = [0] + [np.count_nonzero(Y[:,-1]== i) for i in range(self.n_level)] 
        num_h_Y = np.cumsum(num_h_Y)  
        Y = np.copy(Y)[:,:-1]
        K = np.zeros((X.shape[0], Y.shape[0]))

        for k in range(self.n_level):
            K_k = np.zeros((X.shape[0], Y.shape[0]))
            K_k[num_h_X[k]:, num_h_Y[k]:] = self.rbfs[k](X[num_h_X[k]:,:], 
                                                         Y[num_h_Y[k]:,:])
            for i in range(k, self.n_level):
                for j in range(k, self.n_level):
                    scale1 = np.prod(rhos[k:i]) 
                    scale2 = np.prod(rhos[k:j]) 
                    K_k[num_h_X[i]: num_h_X[i+1],
                            num_h_Y[j] : num_h_Y[j+1]] *= scale1 * scale2
            K += K_k
        return K

    def diag(self, X): 
        
        rhos = [np.log(exp_rho.constant_value) for exp_rho in self.exp_rhos]
        num_h_X = [0] + [np.count_nonzero(X[:,-1]== i) for i in range(self.n_level)]   
        num_h_X = np.cumsum(num_h_X)
        X = np.copy(X)[:,:-1]

        # Generate diagonal of the kernel matrix
        K_diag = np.zeros(X.shape[0])
        for k in range(self.n_level): # Every level 
            K_k_diag = np.zeros(X.shape[0])
            K_k_diag[num_h_X[k]:] += self.rbfs[k].diag(X[num_h_X[k]:,:])
            for i in range(k, self.n_level):
                scale1 = np.prod(rhos[k:i])
                K_k_diag[num_h_X[i]: num_h_X[i+1]] *= scale1 ** 2
            K_diag += K_k_diag
        return K_diag


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
        rhos = [np.log(exp_rho.constant_value) for exp_rho in self.exp_rhos]
        num_h_X1 = [0] + [np.count_nonzero(X1[:,-1]== i) 
                          for i in range(self.n_level)]   
        num_h_X2 = [0] + [np.count_nonzero(X2[:,-1]== i) 
                          for i in range(self.n_level)]  
        num_h_X1, num_h_X2 = np.cumsum(num_h_X1), np.cumsum(num_h_X2)
        X1, X2 = np.copy(X1)[:,:-1], np.copy(X2)[:,:-1]
        
        res = np.zeros((X1.shape[0], X2.shape[0]))
        for i in range(self.n_level): # The highest level
            for j in range(self.n_level):
                # for each block, compute the double summation. 
                for l in range(i + 1):
                    for r in range(j + 1):
                        res[num_h_X1[i]: num_h_X1[i+1],
                            num_h_X2[j]: num_h_X2[j+1]] += \
                            (self.rbfs[l].intK1K2Norm(X1[num_h_X1[i]:num_h_X1[i+1],:],
                                                      X2[num_h_X2[j]:num_h_X2[j+1],:],
                                                self.rbfs[r], mean, cov) 
                                  * np.prod(rhos[l:i]) * np.prod(rhos[l:])
                                  * np.prod(rhos[r:j]) * np.prod(rhos[r:])) 
                        #print(i, j, l, r)
        return res 

    def gradient_x(self, x, X):
        '''compute d cov(f_i(pos), Y(X)) d pos
        '''
        rhos = [np.log(exp_rho.constant_value) for exp_rho in self.exp_rhos]
        h_x = int(x[0][-1]) # level of x
        num_h_X = [0] + [np.count_nonzero(X[:,-1]== i) 
                          for i in range(self.n_level)]  
        num_h_X = np.cumsum(num_h_X)
        x, X = np.copy(x)[:,:-1], np.copy(X)[:,:-1] # data without level
        res = np.zeros((X.shape[0], x.reshape(-1).shape[0])) # (n, d)

        for k in range(h_x+1):
            res_k = np.zeros((X.shape[0], x.reshape(-1).shape[0]))
            res_k[num_h_X[k]:] = self.rbfs[k].gradient_x(x, X[num_h_X[k]:])
            for j in range(k, self.n_level):
                scale1 = np.prod(rhos[k:h_x]) 
                scale2 = np.prod(rhos[k:j]) 
                res_k[num_h_X[j]: num_h_X[j+1]] *= scale1 * scale2
            res += res_k
        return res


    def dintKKNorm_dx(self, x, X, mean, cov):
        '''Compute d \int cov(f_i(pos), x')cov(x', Y)N(x';mean,cov)dx' d pos
        '''
        rhos = [np.log(exp_rho.constant_value) for exp_rho in self.exp_rhos]
        h_x = int(x[0][-1]) # level of x
        num_h_X = [0] + [np.count_nonzero(X[:,-1]== i) 
                          for i in range(self.n_level)]  
        num_h_X = np.cumsum(num_h_X)
        x, X = np.copy(x)[:,:-1], np.copy(X)[:,:-1] # data without level
        
        res = np.zeros((X.shape[0], x.reshape(-1).shape[0])) # (n, d)
        for j in range(self.n_level):
            for l in range(h_x + 1):
                for r in range(j + 1):
                    res[num_h_X[j]: num_h_X[j+1], :] += \
                        (self.rbfs[l].dintK1K2Norm_dx(x, 
                                                X[num_h_X[j]:num_h_X[j+1],:],
                                                self.rbfs[r], mean, cov) 
                                * np.prod(rhos[l:h_x]) * np.prod(rhos[l:])
                                * np.prod(rhos[r:j]) * np.prod(rhos[r:])) 
        return res

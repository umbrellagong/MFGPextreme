import numpy as np
from scipy.linalg import cho_solve, inv
from scipy.stats import norm
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.mixture import GaussianMixture as GMM
from .utils import custom_KDE
import time

class Acq(object):
    '''
    The base acq class.
    '''
    def __init__(self, inputs):
        self.inputs = inputs

    def compute_value(self, x):
        raise NotImplementedError

    def update_prior_search(self, model):     
        raise NotImplementedError


class AcqLW(Acq): 
    ''' Select the next sample for estimating extreme event statistics. 

    This acquisition can be used in both single and multi-fidelity contexts.

    parameters:
    ---------
    inputs: instance of Input class
        Input of the problem including pdf information and a sampling method. 
    ll_type: string
        the type of the weights, must be one of
        (1) rare: w(x)=p(x)/p(y(x))
        (2) extreme: w(x)=p(x)|y(x)-z|^n   
        (3) plain: no weights
        (4) input: w(x)=p(x)               
    load_pts: bool
        whether load the input samples from a txt file
    ll_kwargs: key words for extreme ll_type

    attributes:
    ----------
    model: instance of gpr.GaussianProcessRegressor 
        The surrogate model based on current dataset
    DX: array
        The inputs of current samples
    gmm: instance of sklearn.GMM
        The gmm to approximate likelihood, including gmm.means_, 
        gmm.covariances_, and gmm.scores_.
    '''
    
    def __init__(self, inputs, ll_type='rare', load_pts=False, **ll_kwargs): 
        self.inputs = inputs
        self.ll_type = ll_type
        self.load_pts = load_pts
        self.ll_kwargs = ll_kwargs
        if load_pts:
            smpl = np.loadtxt('map_samples.txt')
            self.pts = smpl[:,0:-1]  # mc points
            self.fx = smpl[:,-1]     # pdf of mc points

    def compute_value_tf_cost(self, pos, fidelity, cost):
        ''' Compute the benefit per cost of adding a sample (pos, fidelity)
        '''
        x = np.append(pos, fidelity)
        value, gradient = self.compute_value(x)
        return value/cost, gradient/cost
        
    def compute_value(self, x):
        ''' Compute the benefit of adding a sample x
        
        For single fidelity, x = pos, while for multi-fidelity, 
        x = {pos, fidelity}.
        '''
        x = np.atleast_2d(x)
        integral, integral_derivative = self.compute_integral(x)
        cov, cov_deriv = self.model.post_cov(x)
        
        value = (integral / cov).item()
        gradient = 1/cov**2 * (cov*integral_derivative - integral*cov_deriv)
        gradient = gradient.reshape(-1)
        return -value, -gradient

    def compute_integral(self, x):
        ''' \int cov^2(f_i(pos), f_h(x'))*w(x')dx', x = {pos, i=fidelity}
        Eq.(15) in paper. 
                                    and 
        d \int cov^2(f_i(pos), f_h(x'))*w(x')dx' d pos, 
            x = {pos, i=fidelity} Eq.(49) in paper.
        '''
        # compute value
        kernel = self.model.kernel_
        integral = self.compute_mixed_kappa(x,x)
        alpha = cho_solve((self.model.L_, True), kernel(self.X, x))
        integral += alpha.T.dot(np.dot(self.kappaXX, alpha) 
                                - 2*self.compute_mixed_kappa(self.X, x))
        # compute derivative
        term1 = 2*self.compute_mixed_dkappa_dx(x,x)
        dalpha_dx = cho_solve((self.model.L_, True), 
                               kernel.gradient_x(x, self.X))
        term2 = 2 * alpha.T.dot(np.dot(self.kappaXX, dalpha_dx))
        term3 = 2 * alpha.T.dot(self.compute_mixed_dkappa_dx(x,self.X))
        term3 += 2 * self.compute_mixed_kappa(x, self.X).dot(dalpha_dx)
                                
        return integral, term1 + term2 - term3
        
    
    def update_prior_search(self, model):
        ''' Update the model(gpr), data(X), compute the gmm of weights and 
        kappa(X,X).
        '''
        self.model = model
        self.X = self.model.X_train_
        # generate GMM approximation of the likelihood
        self._prepare_likelihood(self.ll_type, **self.ll_kwargs)
        # constant for all hypothetical point
        self.kappaXX = self.compute_mixed_kappa(self.X, self.X)

    def compute_mixed_kappa(self, X1, X2):
        ''' compute averaged kappa w.r.t gmm components. 

        Eq. (18) in paper. The 'G' function relies on kernel properties. 
        '''
        kernel = self.model.kernel_
        mixed_kappa = 0
        for i in range(self.gmm.n_components): # the number of gmm component
            mixed_kappa += self.gmm.weights_[i] * kernel.intKKNorm(X1, X2,
                                                    self.gmm.means_[i],
                                                    self.gmm.covariances_[i])
        return mixed_kappa

    def compute_mixed_dkappa_dx(self, x, X):
        ''' Compute the averaged kappa derivatives.
        
        Eq.(53) in paper.
        '''
        kernel = self.model.kernel_
        mixed_kappa = 0
        for i in range(self.gmm.n_components): 
            mixed_kappa += self.gmm.weights_[i] * kernel.dintKKNorm_dx(x, X,
                                                    self.gmm.means_[i],
                                                    self.gmm.covariances_[i])
        return mixed_kappa

    def _prepare_likelihood(self, ll_type, n_components=2, power=6, 
                                           center=0, depressed_side=None):
        '''Compute gmm components of w(x').
        '''
        if self.load_pts:
            pts = self.pts
            fx = self.fx
            n_samples = pts.shape[0]
        else:
            if self.inputs.dim <= 2:
                n_samples = int(1e5)  
            else: 
                n_samples = int(1e6)
            pts = self.inputs.sampling(n_samples)  # input-samples
            fx = self.inputs.pdf(pts)              # weights
        
        if ll_type =='input':                 
            w_raw = fx
        elif ll_type == 'plain':
            w_raw = 1
        else:
            # compute the mean prediction for input-samples
            if self.X.shape[1] != self.inputs.dim:
                aug_pts = np.concatenate((pts, [[1]] * n_samples), axis = 1)
            else:
                aug_pts = pts

            if ll_type == 'rare': 
                if n_samples > 4*1e5:
                    aug_pts_list = np.array_split(aug_pts, 10)
                    mu = np.empty(0)
                    for iii in range(10):
                        mu = np.concatenate((mu, 
                               self.model.predict(aug_pts_list[iii]).flatten()))
                else:
                    mu = self.model.predict(aug_pts).flatten()
                    
                x, y = custom_KDE(mu, weights=fx).evaluate()
                self.fy_interp = InterpolatedUnivariateSpline(x, y, k=1)
                w_raw = fx/self.fy_interp(mu)

            elif ll_type == 'extreme':
                mu = self.model.predict(aug_pts).flatten()
                if center == 'mean': 
                    center = np.average(mu, fx)
                if depressed_side == 'negative':
                    w_raw = fx*abs(mu - center) ** (power*np.sign(mu - center))
                elif depressed_side == 'positive':
                    w_raw = fx*abs(mu - center) ** (-power*np.sign(mu - center))
                else:
                    w_raw = fx*abs(mu - center)**power

            elif ll_type == 'failure':
                # P(X)(1-P(X)) * p(X) / var(X)
                mu, std = self.model.predict(aug_pts, return_std=True)
                # failure probability as a Bernoulli RV
                p = norm.cdf(mu.flatten()/std.flatten())   
                vb = p*(1-p) # var of the Bernoulli
                vf = std**2 # var of the predictions
                w_raw = vb * fx / vf
                
        self.gmm = self._fit_gmm(pts, w_raw, n_components)
        return self

    @staticmethod
    def _fit_gmm(pts, w_raw, n_components):
        '''Fit gmm with weighted samples
        '''
        sca = np.sum(w_raw)
        rng = np.random.default_rng()
        aa = rng.choice(pts, size=50000, p=w_raw/sca)
        gmm = GMM(n_components=n_components, covariance_type="full")
        gmm = gmm.fit(X=aa) 
        return gmm
    




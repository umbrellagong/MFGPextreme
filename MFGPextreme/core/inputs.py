import scipy.stats as stats
from pyDOE import lhs


class GaussianInputs: 
    'A class for Gaussian inputs'
    def __init__(self, mean, cov, domain, dim):
        self.mean = mean
        self.cov = cov
        self.domain = domain
        self.dim = dim
    def sampling(self, num, criterion=None):
        lhd = lhs(self.dim, num, criterion=criterion)
        lhd = self.rescale_samples(lhd, self.domain)
        return lhd
    def pdf(self, x):
        return stats.multivariate_normal(self.mean, self.cov).pdf(x) 

    @staticmethod
    def rescale_samples(x, domain):
        """Rescale samples from [0,1]^d to actual domain."""
        for i in range(x.shape[1]):
            bd = domain[i]
            x[:,i] = x[:,i]*(bd[1]-bd[0]) + bd[0]
        return x
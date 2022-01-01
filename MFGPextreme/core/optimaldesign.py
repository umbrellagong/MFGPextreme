import copy
import numpy as np
from scipy import optimize
from sklearn.base import clone
from joblib import Parallel, delayed
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from .metrics import log_pdf


class OptimalDesign(object):
    '''
    Single-fidelity Bayesian experimental design.

    Parameters
    -----------
    f: func
        The black-box function with input as parameter and output as return 
    input: instance of Inputs class
        Include bounds, pdf, sampling pool or GMM approximations.
        
    Attributes
    ----------
    DX : array (n_samples, n_dim)
        The input of the samples.
    DY : array (n_samples,)
        The output of the samples.
    '''

    def __init__(self, f, inputs):
        self.f = f
        self.inputs = inputs

    def init_sampling(self, n_init): 
        '''
        Generate initial samples.

        Parameters
        ----------
        n_init: int
            The number of initial samples.
        '''
        self.DX = self.inputs.sampling(n_init)
        self.DY = self.f(self.DX)   # a vector
        return self

    def seq_sampling(self, n_seq, acq, model, n_starters=30, n_jobs=6, 
                                              jac=True): 
        '''
        Generate sequential samples. 
        
        Parameters
        ----------
        n_seq: int
            The number of sequential samples.
        acq: instance of Acq class
            Represent the design of acq (or the objective of problem).
        model: instance of GaussianProcessRegressor
            The learned Gaussian process regressor from samples.
        n_starters: int
            Number of restarts for the optimizer.
        n_jobs: int
            The number of workers used by joblib for parallel computation.
        jac: bool
            Whether to use derivative information
            
        Return
        ----------
        model_list: list of GaussianProcessRegressor instance
            The trained gpr at each iteration.
        '''
        
        self.acq = copy.copy(acq)
        self.model = clone(model)
        self.model_list = []
        for i in range(n_seq):
            self.model.fit(self.DX, self.DY)
            # The number of samples are assumed to be small, thus we have the 
            # surrogates in memory.
            self.model_list.append(copy.deepcopy(self.model))
            self.acq.update_prior_search(self.model)
            init = self.inputs.sampling(n_starters)
            res = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value,
                                            init[j], method="L-BFGS-B",
                                            bounds = self.inputs.domain,
                                            jac = jac, options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            opt_pos = res[np.argmin([k.fun for k in res])].x
            self.DX = np.append(self.DX, np.atleast_2d(opt_pos), axis=0)
            self.DY = np.append(self.DY, self.f(opt_pos))

        # train the last model
        self.model.fit(self.DX, self.DY)
        self.model_list.append(copy.deepcopy(self.model))

        return self.model_list

################################################################################

class OptimalDesignTF(object):        
    '''
    Multi-fidelity Bayesian optimal design.

    Parameters
    -----------
    f_h, f_l: func
        The high and low fidelity black-box function with input as parameter and
        output as return 
    input: instance of Inputs class
        Include bounds, pdf, sampling pool or GMM approximations.

    Attributes
    ----------
    DX : array (n_samples, n_dim + 1)
        The input of the samples sorted with decreasing fidelity. The last 
        feature is the fidelity level with 1 and 0 respectively representing 
        high and low-fidelity.
    DY : array (n_samples,)
        The output of the samples.
    '''
    
    def __init__(self, f_h, f_l, inputs):
        self.f_h = f_h
        self.f_l = f_l
        self.inputs = inputs

    def load_data(self, DX):
        ''' Start from existing dataset DX.
        '''
        idx_low = np.where(DX[:,2]==0)[0][0]
        DX_h = DX[:idx_low][:,:2]
        DY_h = self.f_h(DX_h)
        DX_l = DX[idx_low:][:,:2]
        DY_l = self.f_l(DX_l)

        
        DX = convert_x_list_to_array([DX_l, DX_h])
        DY = np.append(DY_l, DY_h)
        self.DX = np.flip(DX, axis=0) 
        self.DY = np.flip(DY)
        return self

    def init_sampling(self, n_init_h, n_init_l): 
        '''Generate initial samples.
        
        Parameters
        -----------
        n_init_h, n_init_l: int
            number of high and low-fidelity initial samples
        '''
        DX_h = self.inputs.sampling(n_init_h)
        DY_h = self.f_h(DX_h)

        DX_l = self.inputs.sampling(n_init_l)
        DY_l = self.f_l(DX_l)

        DX = convert_x_list_to_array([DX_l, DX_h])
        DY = np.append(DY_l, DY_h)
        self.DX = np.flip(DX, axis=0) 
        self.DY = np.flip(DY)

        return self

    def seq_sampling_fixed(self, n_seq, n_ratio, acq, model, 
                     n_starters=30, n_jobs=6): 
        '''Generate sequential samples corresponding to MF-F.
        
        Parameters
        -----------
        n_seq: int
            number of sequential iterations (including one high-fidelity and 
            n_ratio low-fidelity samples).
        n_ratio: int
            ratio of low/high-fidelity samples in each iteration.
        acq, model, n_starters, n_jobs: see OptimalDesign
        
        Return
        ----------
        model_list: list of GaussianProcessRegressor instance
            The trained gpr at each iteration.
        '''
        self.acq = copy.deepcopy(acq)
        self.model = copy.deepcopy(model)
        self.model_list = []

        for ii in range(n_seq):
            fidelity = 1
            for j in range(n_ratio+1):
                self.model.fit(self.DX, self.DY)
                self.model_list.append(copy.deepcopy(self.model)) 
                self.acq.update_prior_search(self.model)
                init = self.inputs.sampling(n_starters)
                res = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                                (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(fidelity,1),
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
                self.res = res   
                opt_pos = res[np.argmin([k.fun for k in res])].x
                
                if fidelity==1:
                    self.DX = np.insert(self.DX, 0, 
                                        np.append(opt_pos, 1), axis=0)
                    self.DY = np.insert(self.DY, 0, self.f_h(opt_pos))      
                    fidelity = 0
                else:
                    self.DX = np.insert(self.DX, self.DX.shape[0], 
                                        np.append(opt_pos, 0), axis=0)
                    self.DY = np.append(self.DY, self.f_l(opt_pos))
        self.model.fit(self.DX, self.DY)
        self.model_list.append(copy.deepcopy(self.model))
        return self.model_list

    def seq_sampling_opt(self, n_seq, n_cost, c_ratio, acq, model, 
                     n_starters=12, n_jobs=6): 
        '''Generate sequential samples corresponding to MF-O.
        
        Parameters
        -----------
        n_seq: int
            number of sequential samples. 
        n_cost: float
            cost limit
        c_ratio: float
            ratio of high/low-fidelity costs
        acq, model, n_starters, n_jobs: see OptimalDesign
        
        Return
        ----------
        model_list: list of GaussianProcessRegressor instance
            The trained gpr at each iteration.
        '''
        
        self.acq = copy.deepcopy(acq)
        self.model = copy.deepcopy(model)
        self.model_list = []

        for ii in range(n_seq):
            self.model.fit(self.DX, self.DY)
            self.model_list.append(copy.deepcopy(self.model))
            self.acq.update_prior_search(self.model)
            init = self.inputs.sampling(n_starters)
            res_l = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(0, 1), # low-fidelity
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_l = res_l  
            opt_pos_l = res_l[np.argmin([k.fun for k in res_l])].x
            opt_value_l = res_l[np.argmin([k.fun for k in res_l])].fun
            res_h = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(1, c_ratio),
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_h = res_h  
            opt_pos_h = res_h[np.argmin([k.fun for k in res_h])].x
            opt_value_h = res_h[np.argmin([k.fun for k in res_h])].fun
            
            if opt_value_h < opt_value_l: # high-fidelity sampling!
                self.DX = np.insert(self.DX, 0, 
                                    np.append(opt_pos_h, 1), axis=0)
                self.DY = np.insert(self.DY, 0, self.f_h(opt_pos_h))      
            else:
                self.DX = np.insert(self.DX, self.DX.shape[0], 
                                    np.append(opt_pos_l, 0), axis=0)
                self.DY = np.append(self.DY, self.f_l(opt_pos_l))
            
            num_h_X = np.count_nonzero(self.DX[:,-1]==1)
            num_l_X = np.count_nonzero(self.DX[:,-1]==0)
            cost = num_h_X + 1 / c_ratio * num_l_X
            if cost > n_cost:
                break
            
        self.model.fit(self.DX, self.DY)
        self.model_list.append(copy.deepcopy(self.model))
        return self.model_list



########################################################


class OptimalDesignTF_light(OptimalDesignTF): 
    ''' Same as the father class.
    '''
    
    def init_sampling(self, n_init_h, n_init_l, seed=0): 
        '''Generate initial samples.
        
        Parameters
        -----------
        n_init_h, n_init_l: int
            number of high and low-fidelity initial samples
        seed: int,
            random seed
        '''
        self.n_init_h = n_init_h
        self.n_init_l = n_init_l
        self.seed = seed
        np.random.seed(seed)
        DX_h = self.inputs.sampling(n_init_h)
        DX_l = self.inputs.sampling(n_init_l)
        DY_h = self.f_h(DX_h)
        DY_l = self.f_l(DX_l)
        
        DX = convert_x_list_to_array([DX_l, DX_h])
        DY = np.append(DY_l, DY_h)
        self.DX = np.flip(DX, axis=0) 
        self.DY = np.flip(DY)
        return self

    def seq_sampling_fixed(self, n_seq, n_ratio, n_cost, c_ratio, acq, model, 
                     n_starters=12, n_jobs=6, opt_hyper_threshold=500, 
                     validation=True, FILENAME='now', **metric_kwargs):
        '''Generate sequential samples corresponding to MF-F.
        
        The results, in terms of cost and errors, are directly saved to 
        FILENAME. The trained model and dataset will not be saved. 
        
        Parameters
        -----------
        n_seq: int
            number of sequential iterations (including one high-fidelity and 
            n_ratio low-fidelity samples).
        n_ratio: int
            ratio of low/high-fidelity samples in each iteration.
        n_cost: float
            cost limit
        c_ratio: float
            ratio of high/low-fidelity costs
        opt_hyper_threshold: float
            The hyper-parameters of the surrogate will be optimized every 5 
            samples after the total cost exceeding this limit.
        validation: bool
            If it is a validation case, the error w.r.t. exact pdf will be 
            computed.
        FILENAME: str
            The name of folder to save computed errors in validation case.
        metric_kwargs: 
            keyword parameters to compute the error.
        '''
        self.acq = copy.deepcopy(acq)
        self.model = copy.deepcopy(model)
        cost = self.n_init_h + self.n_init_l/c_ratio

        for ii in range(n_seq):
            fidelity = 1
            for j in range(n_ratio+1):
                if self.DX.shape[0]<opt_hyper_threshold or self.DX.shape[0]%5==0:
                    self.model.fit(self.DX, self.DY)
                else:
                    self.model.fit_keep(self.DX, self.DY)
                if validation:
                    # writedown the error if we have a exact pdf.
                    error = np.log10(log_pdf([self.model], **metric_kwargs)[0])
                    result = np.array([[ii, cost, error]])
                    with open(FILENAME + '/' + str(self.seed) +'.out', 'a') as f:
                            np.savetxt(f, result, fmt='%1.4f')

                self.acq.update_prior_search(self.model)
                init = self.inputs.sampling(n_starters)
                res = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                                (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(fidelity,1),
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
                self.res = res   
                opt_pos = res[np.argmin([k.fun for k in res])].x
                
                if fidelity==1:
                    self.DX = np.insert(self.DX, 0, 
                                        np.append(opt_pos, 1), axis=0)
                    self.DY = np.insert(self.DY, 0, self.f_h(opt_pos))
                    cost = cost + 1
                    # the MF-F assume one high-fidelity in each iteration
                    fidelity = 0
                else:
                    self.DX = np.insert(self.DX, self.DX.shape[0], 
                                        np.append(opt_pos, 0), axis=0)
                    self.DY = np.append(self.DY, self.f_l(opt_pos))
                    cost = cost +  1 / c_ratio
        if validation:
            self.model.fit(self.DX, self.DY)
            error = np.log10(log_pdf([self.model], **metric_kwargs)[0])
            result = np.array([[ii, cost, error]])
            with open(FILENAME + '/' + str(self.seed) +'.out', 'a') as f:
                        np.savetxt(f, result)
        return None

    def seq_sampling_opt(self, n_seq, n_ratio, n_cost, c_ratio, acq, model, 
                     n_starters=12, n_jobs=6, opt_hyper_threshold=500,
                     validation=True, FILENAME='now', **metric_kwargs): 
        '''Same as seq_sampling_fixed.
        '''
        self.acq = copy.deepcopy(acq)
        self.model = copy.deepcopy(model)
        cost = self.n_init_h + self.n_init_l/c_ratio

        for ii in range(n_seq):
            if self.DX.shape[0]<opt_hyper_threshold or self.DX.shape[0]%5==0:
                self.model.fit(self.DX, self.DY)
            else:
                self.model.fit_keep(self.DX, self.DY)
            if validation:
                error = np.log10(log_pdf([self.model], **metric_kwargs)[0])
                result = np.array([[ii, cost, error]])
                with open(FILENAME + '/' + str(self.seed) +'.out', 'a') as f:
                        np.savetxt(f, result, fmt='%1.4f')

            self.acq.update_prior_search(self.model)
            init = self.inputs.sampling(n_starters)
            # compute the low-fidelity sub-optimal sample
            res_l = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(0, 1), # low-fidelity
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_l = res_l  
            opt_pos_l = res_l[np.argmin([k.fun for k in res_l])].x
            opt_value_l = res_l[np.argmin([k.fun for k in res_l])].fun
            # compute the high-fidelity sub-optimal sample
            res_h = Parallel(n_jobs=n_jobs)(delayed(optimize.minimize)
                                            (self.acq.compute_value_tf_cost,
                                                init[j], 
                                                args=(1, c_ratio),
                                                method="L-BFGS-B",
                                                jac=True,
                                                bounds = self.inputs.domain,
                                                options={'gtol': 1e-3})
                                            for j in range(init.shape[0]))
            self.res_h = res_h  
            opt_pos_h = res_h[np.argmin([k.fun for k in res_h])].x
            opt_value_h = res_h[np.argmin([k.fun for k in res_h])].fun

            if opt_value_h < opt_value_l: # high-fidelity sampling!
                self.DX = np.insert(self.DX, 0, 
                                    np.append(opt_pos_h, 1), axis=0)
                self.DY = np.insert(self.DY, 0, self.f_h(opt_pos_h))
                cost = cost + 1      
            else:
                self.DX = np.insert(self.DX, self.DX.shape[0], 
                                    np.append(opt_pos_l, 0), axis=0)
                self.DY = np.append(self.DY, self.f_l(opt_pos_l))
                cost = cost +  1 / c_ratio
            if cost > n_cost:
                break
        if validation:
            self.model.fit(self.DX, self.DY)
            error = np.log10(log_pdf([self.model], **metric_kwargs)[0])
            result = np.array([[ii, cost, error]])
            with open(FILENAME + '/' + str(self.seed) +'.out', 'a') as f:
                        np.savetxt(f, result)
        return None
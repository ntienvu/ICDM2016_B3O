from __future__ import division
import numpy as np
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
#from prada_gaussian_process import PradaGaussianProcess


class AcquisitionFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind='nei', kappa=1.96, L=1,k_Neighbor=2):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        self.L=L
        self.k_Neighbor=k_Neighbor
        
        if kind not in ['bucb','ucb', 'ei', 'poi','nei','lei']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def acq_kind(self, x, gp, y_max):
        
        #print self.kind
        if np.any(np.isnan(x)):
            return 0
        if self.kind == 'bucb':
            return self._bucb(x, gp, self.kappa)
            
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            #return self._ei(x, gp, y_max)
            return self._ei(x, gp, y_max)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max)

    def utility_plot(self, x, gp, y_max):
        
        if np.any(np.isnan(x)):
            return 0

        if self.kind == 'ei':
            return self._ei_plot(x, gp, y_max)
 
        
    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _bucb(x, gp, kappa):
        mean, var = gp.predict_bucb(x, eval_MSE=True)
        var.flags['WRITEABLE']=True
        #var=var.copy()
        var[var<1e-10]=0
        mean=np.atleast_2d(mean).T
        var=np.atleast_2d(var).T
                
        return mean + kappa * np.sqrt(var)
        
    @staticmethod
    def _ei(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)
        
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
 
            
            z = (mean - y_max)/np.sqrt(var)
        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])

            
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
                
            if len(x)==1000:    
                return out
            else:
                return np.mean(out)
                    
    
	
        
                
    # for plot purpose
    @staticmethod
    def _ei_plot(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)
        
        if gp.nGP==0:
            var = np.maximum(var, 1e-9 + 0 * var)
    
            #mean=np.mean(mean)
            #var=np.mean(var)
            
            z = (mean - y_max)/np.sqrt(var)
        
            out=(mean - y_max) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)
            
            return out
        else:                
            z=[None]*gp.nGP
            out=[None]*gp.nGP
            # Avoid points with zero variance
            for idx in range(gp.nGP):
                var[idx] = np.maximum(var[idx], 1e-9 + 0 * var[idx])
    
            #mean=np.mean(mean)
            #var=np.mean(var)
            
                z[idx] = (mean[idx] - y_max)/np.sqrt(var[idx])
            
                out[idx]=(mean[idx] - y_max) * norm.cdf(z[idx]) + np.sqrt(var[idx]) * norm.pdf(z[idx])
        

            out=np.asarray(out)
            return np.mean(out,axis=0)
      
            
    @staticmethod
    def _poi(x, gp, y_max):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max)/np.sqrt(var)
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]
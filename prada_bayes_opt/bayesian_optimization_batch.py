# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
from prada_gaussian_process import *
from prada_bayes_opt import *
from prada_bayes_opt import visualization
from acquisition_maximization import *

from sklearn.metrics.pairwise import euclidean_distances
from sklearn import cluster
from sklearn import mixture
import matplotlib.pyplot as plt
import time

#import nlopt


#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptBatch(object):

    def __init__(self, f, pbounds, acq='ucb', verbose=1, opt='scipy'):
        """      
        Input parameters
        ----------
        f:              function to optimize:        
        pbounds:        bounds on parameters        
        acq:            acquisition function, 'ei', 'ucb'        
        opt:            optimization toolbox, 'nlopt','direct','scipy'
        
        Returns
        -------
        dim:            dimension
        bounds:         bounds on original scale
        scalebounds:    bounds on normalized scale of 0-1
        time_opt:       will record the time spent on optimization
        gp:             Gaussian Process object
        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds

        if isinstance(pbounds,dict):
            # Get the name of the parameters
            self.keys = list(pbounds.keys())
        
            self.bounds = []
            for key in self.pbounds.keys():
                self.bounds.append(self.pbounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(pbounds)

        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]

        # Some function to be optimized
        self.f = f

        self.opt=opt
        # acquisition function type
        self.acq=acq
        
        # store the batch size for each iteration
        self.NumPoints=[]
        # Numpy array place holders
        self.X_original= None
        
        # scale the data to 0-1 fit GP better
        self.X = None # X=( X_original - min(bounds) / (max(bounds) - min(bounds))
        self.Y = None
        self.opt_time=[]
        
        self.L=0 # lipschitz

        self.gp=PradaGaussianProcess

        # Acquisition Function
        self.acq_func = None
        
    def posterior(self, Xnew):
        #xmin, xmax = -2, 10
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, n_init_points):
        """      
        Input parameters
        ----------
        gp_params:            Gaussian Process structure      
        n_init_points:        # init points
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=n_init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        temp=np.asarray(l)
        temp=temp.T
        init_X=list(temp.reshape((n_init_points,-1)))

        # Evaluate target function at all initialization           
        y_init=self.f(init_X)

        # Turn it into np array and store.
        self.X_original=np.asarray(init_X)
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X_original = np.asarray(init_X)
        self.X = np.asarray(temp_init_point)
        y_init=np.reshape(y_init,(n_init_points,1))
        self.Y = np.asarray(y_init)
        
        self.NumPoints=np.append(self.NumPoints,n_init_points)
        
        print "#Batch={:d} f_max={:.4f}".format(n_init_points,self.Y.max())
   

    def fitIGMM(self,obs,IsPlot=0):
        """
        Fitting the Infinite Gaussian Mixture Model and GMM where applicable
        Input Parameters
        ----------
        
        obs:        samples  generated under the acqusition function by BGSS
        
        IsPlot:     flag variable for visualization    
        
        
        Returns
        -------
        mean vector: mu_1,...mu_K
        """

        if self.dim<=2:
            n_init_components=3
        else:
            n_init_components=np.int(self.dim*1.1)
            
        dpgmm = mixture.DPGMM(n_components=n_init_components,covariance_type="full",min_covar=1e-3)
        dpgmm.fit(obs) 

        # check if DPGMM fail, then use GMM.
        mydist=euclidean_distances(dpgmm.means_,dpgmm.means_) 
        np.fill_diagonal(mydist,99)

        if dpgmm.converged_ is False or np.min(mydist)<(0.01*self.dim):
            dpgmm = mixture.GMM(n_components=n_init_components,covariance_type="full",min_covar=1e-5)
            dpgmm.fit(obs)  

        # truncated for variational inference
        weight=dpgmm.weights_
        weight_sorted=np.sort(weight)
        weight_sorted=weight_sorted[::-1]
        temp_cumsum=np.cumsum(weight_sorted)
        
        cutpoint=0
        for idx,val in enumerate(temp_cumsum):
            if val>0.7:
                cutpoint=weight_sorted[idx]
                break
        
        ClusterIndex=[idx for idx,val in enumerate(dpgmm.weights_) if val>=cutpoint]        
               
        myMeans=dpgmm.means_[ClusterIndex]
        #dpgmm.means_=dpgmm.means_[ClusterIndex]
        dpgmm.truncated_means_=dpgmm.means_[ClusterIndex]
                
        if IsPlot==1 and self.dim<=2:
            visualization.plot_histogram(self,obs)
            visualization.plot_mixturemodel(dpgmm,self,obs)

        new_X=myMeans.reshape((len(ClusterIndex), -1))
        new_X=new_X.tolist()
        
        return new_X
    
    def maximize_batch_B3O(self,gp_params, kappa=2,IsPlot=0):
        """
        Finding a batch of points using Budgeted Batch Bayesian Optimization approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        Returns
        -------
        X: a batch of [x_1..x_Nt]
        """

                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)
        
        
        # Step 2 in the Algorithm
        
        # Set parameters for Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        # record optimization time
        start_gmm_opt=time.time()
        
        
        if IsPlot==1 and self.dim<=2:#plot
            visualization.plot_bo(self)                
                
        # Step 4 in the Algorithm
        # generate samples from Acquisition function
        
        # check the bound 0-1 or original bound        
        obs=acq_batch_generalized_slice_sampling_generate(self.acq_func.acq_kind,self.gp,self.scalebounds,N=500,y_max=self.Y.max())
        
        
        # Step 5 and 6 in the Algorithm
        if len(obs)==0: # monotonous acquisition function
            print "Monotonous acquisition function"
            new_X=np.random.uniform(self.bounds[:, 0],self.bounds[:, 1],size=self.bounds.shape[0])
            new_X=new_X.reshape((1,-1))
            new_X=new_X.tolist()

        else:
            new_X=self.fitIGMM(obs,IsPlot)
            

        # Test if x_max is repeated, if it is, draw another one at random
        temp_new_X=[]
        for idx,val in enumerate(new_X):
            if np.all(np.any(np.abs(self.X-val)>0.02,axis=1)): # check if a data point is already taken
                temp_new_X=np.append(temp_new_X,val)
                
        
        if len(temp_new_X)==0:
            temp_new_X=np.zeros((1,self.dim))
            for idx in range(0,self.dim):
                temp_new_X[0,idx]=np.random.uniform(self.scalebounds[idx,0],self.scalebounds[idx,1],1)
        else:
            temp_new_X=temp_new_X.reshape((-1,self.dim))
            
        self.NumPoints=np.append(self.NumPoints,temp_new_X.shape[0])


        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))
        
       
        self.X=np.vstack((self.X, temp_new_X))
        
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(temp_new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        
        
        # Step 7 in the algorithm
        # Evaluate y=f(x)
        
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        
        # Step 8 in the algorithm
        
        self.Y=np.append(self.Y,temp)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))

        print "#Batch={:d} f_max={:.3f}".format(temp_new_X.shape[0],self.Y.max())
        
        
        #ur = unique_rows(self.X)
        #self.gp.fit(self.X[ur], self.Y[ur])
        #self.gp.fit_incremental(temp_new_X, temp_new_Y)
        
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

    def maximize_batch_BUCB(self,gp_params, B=5,kappa=2):
        """
        Finding a batch of points using GP-BUCB approach
        
        Input Parameters
        ----------

        gp_params:          Parameters to be passed to the Gaussian Process class
        
        B:                  fixed batch size for all iteration
        
        kappa:              constant value in UCB
        
        IsPlot:             flag variable for visualization    
        
        
        Returns
        -------
        X: a batch of [x_1..x_B]
        """
        
        self.B=B
                
        # Set acquisition function
        self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)
        

               
        # Set parameters if any was passed
        self.gp=PradaGaussianProcess(gp_params)
        
        if len(self.gp.KK_x_x_inv)==0: # check if empty
            self.gp.fit(self.X, self.Y)
        #else:
            #self.gp.fit_incremental(self.X[ur], self.Y[ur])
        
        start_gmm_opt=time.time()
        # generate samples from Acquisition function
        
        y_max=self.gp.Y.max()
        # check the bound 0-1 or original bound        
        temp_X=self.X
        temp_gp=self.gp  
        temp_gp.X_bucb=temp_X
        temp_gp.KK_x_x_inv_bucb=self.gp.KK_x_x_inv
        
        # finding new X
        new_X=[]
        for ii in range(B):
            # Finding argmax of the acquisition function.
            x_max = acq_max(ac=self.acq_func.acq_kind, gp=temp_gp, y_max=y_max, bounds=self.scalebounds)
                     

            if np.any((temp_X - x_max).sum(axis=1) == 0) | np.isnan(x_max.sum()):
                x_max = np.random.uniform(self.scalebounds[:, 0],
                                          self.scalebounds[:, 1],
                                          size=self.scalebounds.shape[0])
                                          
            if ii==0:
                new_X=x_max
            else:
                new_X= np.vstack((new_X, x_max.reshape((1, -1))))
                            
            # update the Gaussian Process and thus the acquisition function                         
            temp_gp.compute_incremental_var(temp_X,x_max)

            temp_X = np.vstack((temp_X, x_max.reshape((1, -1))))
            temp_gp.X_bucb=temp_X
        
        
        # record the optimization time
        finished_gmm_opt=time.time()
        elapse_gmm_opt=finished_gmm_opt-start_gmm_opt
        
        self.opt_time=np.hstack((self.opt_time,elapse_gmm_opt))

        self.NumPoints=np.append(self.NumPoints,B)


        self.X=temp_X
                    
        # convert back to original scale
        temp_X_new_original=[val*self.max_min_gap+self.bounds[:,0] for idx, val in enumerate(new_X)]
        temp_X_new_original=np.asarray(temp_X_new_original)
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        
        # evaluate y=f(x)
        temp=self.f(temp_X_new_original)
        temp=np.reshape(temp,(-1,1))
        self.Y=np.append(self.Y,temp)
            
        print "#Batch={:d} f_max={:.4f}".format(new_X.shape[0],self.Y.max())               

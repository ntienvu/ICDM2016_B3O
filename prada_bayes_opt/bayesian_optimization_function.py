# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:49:58 2016

"""


from __future__ import division
import numpy as np
#from sklearn.gaussian_process import GaussianProcess
from scipy.optimize import minimize
from acquisition_functions import AcquisitionFunction, unique_rows
#from visualization import Visualization
from prada_gaussian_process import PradaGaussianProcess
from prada_gaussian_process import PradaMultipleGaussianProcess

from acquisition_maximization import acq_max_nlopt
from acquisition_maximization import acq_max_direct
from acquisition_maximization import acq_max
from sklearn.metrics.pairwise import euclidean_distances
import time
#import nlopt

#@author: Vu

#======================================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBayOptFn(object):

    def __init__(self, f, pbounds, acq='ei', verbose=1, opt='nlopt'):
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

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        if isinstance(pbounds,dict):
            # Get the name of the parameters
            self.keys = list(pbounds.keys())
        
            self.bounds = []
            for key in pbounds.keys():
                self.bounds.append(pbounds[key])
            self.bounds = np.asarray(self.bounds)
        else:
            self.bounds=np.asarray(pbounds)

        # create a scalebounds 0-1
        scalebounds=np.array([np.zeros(self.dim), np.ones(self.dim)])
        self.scalebounds=scalebounds.T
        
        self.max_min_gap=self.bounds[:,1]-self.bounds[:,0]
        
        # Some function to be optimized
        self.f = f
        # optimization toolbox
        self.opt=opt
        # acquisition function type
        self.acq=acq
        
        # store X in original scale
        self.X_original= None

        # store X in 0-1 scale
        self.X = None
        
        # store y=f(x)
        self.Y = None
        
        self.time_opt=0


        self.k_Neighbor=2
        
        # Lipschitz constant
        self.L=0
        
        
        # Gaussian Process class
        self.gp=PradaGaussianProcess

        # acquisition function
        self.acq_func = None
    
    # will be later used for visualization
    def posterior(self, Xnew):
        self.gp.fit(self.X, self.Y)
        mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
        return mu, np.sqrt(sigma2)
    
    
    def init(self, gp_params, n_init_points=3):
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
        
        self.X_original = np.asarray(init_X)
        
        # Evaluate target function at all initialization           
        y_init=self.f(init_X)
        y_init=np.reshape(y_init,(n_init_points,1))
        self.Y = np.asarray(y_init)
        
        # convert it to scaleX
        temp_init_point=np.divide((init_X-self.bounds[:,0]),self.max_min_gap)
        
        self.X = np.asarray(temp_init_point)
           
    def estimate_L(self,bounds):
        '''
        Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
        '''
        def df(x,model,x0):
            mean_derivative=gp_model.predictive_gradient(self.X,self.Y,x)
            
            temp=mean_derivative*mean_derivative
            if len(temp.shape)<=1:
                res = np.sqrt( temp)
            else:
                try:
                    res = np.sqrt(np.sum(temp,axis=1)) # simply take the norm of the expectation of the gradient        
                except:
                    print "bug"
            return -res

        gp_model=self.gp
                
        dim = len(bounds)
        num_data=100*dim
        samples = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): samples[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)

        #samples = np.vstack([samples,gp_model.X])
        pred_samples = df(samples,gp_model,0)
        x0 = samples[np.argmin(pred_samples)]
        
        #print x0
        res = minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (gp_model,x0), options = {'maxiter': 100})
        
        try:
            minusL = res.fun[0][0]
        except:
            if len(res.fun.shape)==1:
                minusL = res.fun[0]
            else:
                minusL = res.fun
        
        L = -minusL
        
        if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
        return L    
    
    def maximize(self,gp_params,kappa=2):
        """
        Main optimization method.

        Input parameters
        ----------

        kappa: parameter for UCB acquisition only.

        gp_params: parameter for Gaussian Process

        Returns
        -------
        x: recommented point for evaluation
        """

        # init a new Gaussian Process
        self.gp=PradaGaussianProcess(gp_params)
        
        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])
        

        # Set acquisition function
        start_opt=time.time()

        acq=self.acq
        
        # select the acquisition function
        if acq=='nei':
            self.L=self.estimate_L(self.bounds)
            self.util = AcquisitionFunction(kind=self.acq, L=self.L)
        else:
            if acq=="ucb":
                self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)
            else:
                self.acq_func = AcquisitionFunction(kind=self.acq, kappa=kappa)

        y_max = self.Y.max()
        
        # select the optimization toolbox        
        if self.opt=='nlopt':
            x_max,f_max = acq_max_nlopt(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
        if self.opt=='scipy':
            x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)
        if self.opt=='direct':
            x_max = acq_max_direct(ac=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,bounds=self.scalebounds)

        # record the optimization time
        finished_opt=time.time()
        elapse_opt=finished_opt-start_opt
        self.time_opt=np.hstack((self.time_opt,elapse_opt))
        
        # Test if x_max is repeated, if it is, draw another one at random
        if np.any((self.X - x_max).sum(axis=1) == 0):

            x_max = np.random.uniform(self.scalebounds[:, 0],
                                      self.scalebounds[:, 1],
                                      size=self.scalebounds.shape[0])
                                     
        # store X                                     
        self.X = np.vstack((self.X, x_max.reshape((1, -1))))

        # compute X in original scale
        temp_X_new_original=x_max*self.max_min_gap+self.bounds[:,0]
        self.X_original=np.vstack((self.X_original, temp_X_new_original))
        # evaluate Y using original X
        self.Y = np.append(self.Y, self.f(temp_X_new_original))
     
#======================================================================================
#======================================================================================================
#======================================================================================================
#======================================================================================================

class PradaBOFn_MulGP(object):

	def __init__(self, f, pbounds, acq='ei', verbose=1):
		"""
		:param f:
			Function to be maximized.

		:param pbounds:
			Dictionary with parameters names as keys and a tuple with minimum
			and maximum values.

		:param verbose:
			Whether or not to print progress.

		"""
		# Store the original dictionary
		self.pbounds = pbounds


		# Find number of parameters
		self.dim = len(pbounds)

		if isinstance(pbounds,dict):
            # Get the name of the parameters
			self.keys = list(pbounds.keys())
		
			self.bounds = []
			for key in self.pbounds.keys():
				self.bounds.append(self.pbounds[key])
			self.bounds = np.asarray(self.bounds)
		else:
			self.bounds=np.asarray(pbounds)

		# Some function to be optimized
		self.f = f

		# acquisition function type
		self.acq=acq

		# Initialization flag
		self.initialized = False

		# Initialization lists --- stores starting points before process begins
		self.init_points = []
		self.x_init = []
		self.y_init = []

		# Numpy array place holders
		self.X = None
		self.Y = None


		# Since scipy 0.16 passing lower and upper bound to theta seems to be
		# broken. However, there is a lot of development going on around GP
		# is scikit-learn. So I'll pick the easy route here and simple specify
		# only theta0.
		#self.gp = GaussianProcess(theta0=np.random.uniform(0.001, 0.05, self.dim),
								  #thetaL=1e-5 * np.ones(self.dim),
								  #thetaU=1e0 * np.ones(self.dim),random_start=30)

		self.gp=PradaMultipleGaussianProcess
		self.theta=[]
		# Utility Function placeholder
		self.util = None


	def posterior(self,Xnew):
		#xmin, xmax = -2, 10
		self.gp.fit(self.X, self.Y)
  
          
		mu, sigma2 = self.gp.predict(Xnew, eval_MSE=True)
		return mu, np.sqrt(sigma2)


	def init(self, init_points):
		# Generate random points
            l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

		# Concatenate new random points to possible existing
		# points from self.explore method.
            temp=np.asarray(l)
            self.init_points=list(temp.reshape((init_points,-1)))

        # Create empty list to store the new values of the function
            y_init = []

            # Evaluate target function at all initialization
            for x in self.init_points:
                #y_init.append(self.f(**dict(zip(self.keys, x))))
                y_init.append(self.f(x))
    
    
    		# Append any other points passed by the self.initialize method (these
    		# also have a corresponding target value passed by the user).
    		self.init_points += self.x_init
    
    		# Append the target value of self.initialize method.
    		y_init += self.y_init
    
    		# Turn it into np array and store.
    		self.X = np.asarray(self.init_points)
    		self.Y = np.asarray(y_init)
    
    		self.n_batch=1
    
    
    		self.initialized = True


	def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 **gp_params):
		"""
		Main optimization method.

		Parameters
		----------
		:param init_points:
			Number of randomly chosen points to sample the
			target function before fitting the gp.

		:param n_iter:
			Total number of times the process is to repeated. Note that
			currently this methods does not have stopping criteria (due to a
			number of reasons), therefore the total number of points to be
			sampled must be specified.

		:param acq:
			Acquisition function to be used, defaults to Expected Improvement.

		:param gp_params:
			Parameters to be passed to the Scikit-learn Gaussian Process object

		Returns
		-------
		:return: Nothing
		"""
		# Reset timer
		#self.plog.reset_timer()

		# Set acquisition function
		self.util = UtilityFunction(kind=self.acq, kappa=kappa)

		# Initialize x, y and find current y_max
		if not self.initialized:
			#if self.verbose:
				#self.plog.print_header()
			self.init(init_points)

		y_max = self.Y.max()

		self.theta=gp_params['theta']

		# Set parameters if any was passed
		#self.gp.set_params(**gp_params)
		self.gp=PradaMultipleGaussianProcess(**gp_params)

		# Find unique rows of X to avoid GP from breaking
		ur = unique_rows(self.X)
		self.gp.fit(self.X[ur], self.Y[ur])

		# Finding argmax of the acquisition function.
		x_max = acq_max(ac=self.acq_func.acq_kind,gp=self.gp,
						y_max=y_max,bounds=self.bounds)

		#print "start acq max nlopt"
		#x_max,f_max = acq_max_nlopt(f=self.acq_func.acq_kind,gp=self.gp,y_max=y_max,
							  #bounds=self.bounds)
		#print "end acq max nlopt"

		# Test if x_max is repeated, if it is, draw another one at random
		# If it is repeated, print a warning
		#pwarning = False
		if np.any((self.X - x_max).sum(axis=1) == 0):
			#print "x max uniform random"

			x_max = np.random.uniform(self.bounds[:, 0],
									  self.bounds[:, 1],
									  size=self.bounds.shape[0])
									
		#print "start append X,Y"
		self.X = np.vstack((self.X, x_max.reshape((1, -1))))
		#self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))
		self.Y = np.append(self.Y, self.f(x_max))


		#print "end append X,Y"
		#print 'x_max={:f}'.format(x_max[0])

		#print "start fitting GP"

		# Updating the GP.
		ur = unique_rows(self.X)
		self.gp.fit(self.X[ur], self.Y[ur])

		#print "end fitting GP"
		# Update maximum value to search for next probe point.
		if self.Y[-1] > y_max:
			y_max = self.Y[-1]
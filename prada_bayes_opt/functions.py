# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:25:02 2016

"""

import numpy as np
from collections import OrderedDict


def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x
    
class functions:
    def plot(self):
        print "not implemented"
        
    
    
class sincos(functions):
    def __init__(self):
        self.input_dim=1
        self.bounds={'x':(-2,12)}
        self.fmin=11
        self.min=0
        self.ismax=1
        self.name='sincos'
    def func(self,x):
        x=np.asarray(x)

        fval=x*np.sin(x)+x*np.cos(2*x)
        return fval*self.ismax

class fourier(functions):
	'''
	Forrester function. 
	
	:param sd: standard deviation, to generate noisy evaluations of the function.
	'''
	def __init__(self,sd=None):
		self.input_dim = 1		
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 4.795 		## approx
		self.fmin = -9.5083483926941064 			## approx
		self.bounds = {'x':(0,10)}
		self.name='sincos'
		self.ismax=-1

	def func(self,X):
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = X*np.sin(X)+X*np.cos(2*X)
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return self.ismax*fval.reshape(n,1) + noise
        
        
class branin(functions):
    def __init__(self):
        self.input_dim=2
        self.bounds=OrderedDict([('x1',(-5,10)),('x2',(-5,10))])
        self.fmin=0.397887
        self.min=[9.424,2.475]
        self.ismax=-1
        self.name='branin'
    #def func(self,x1,x2):
    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        a=1
        b=5.1/(4*np.pi*np.pi)
        c=5/np.pi
        r=6
        s=10
        t=1/(8*np.pi)
        fx=a*(x2-b*x1*x1+c*x1-r)**2+s*(1-t)*np.cos(x1)+s    
        return fx*self.ismax
        
    
class forrester(functions):
	'''
	Forrester function. 
	:param sd: standard deviation, to generate noisy evaluations of the function.
	'''
	def __init__(self):
		self.input_dim = 1		
		self.min = 0.78 		## approx
		self.fmin = -6.03 			## approx
		self.bounds = {'x':(0,1)}
		self.ismax=-1
		self.name='forrester'
            
	def func(self,x):
		x=np.asarray(x)
		fval = ((6*x -2)**2)*np.sin(12*x-4)
		return fval*self.ismax

  
class rosenbrock(functions):
	'''
	rosenbrock function

	:param bounds: the box constraints to define the domain in which the function is optimized.
	:param sd: standard deviation, to generate noisy evaluations of the function.
	'''
	def __init__(self,bounds=None,sd=None):
		self.input_dim = 2
		if bounds == None: self.bounds = OrderedDict([('x1',(-0.5,3)),('x2',(-1.5,2))])
		else: self.bounds = bounds
		self.min = [(0, 0)]
		self.fmin = 0
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.ismax=-1
		self.name = 'Rosenbrock'

	def func(self,X):
		X=np.asarray(X)
		if len(X.shape)==1:
			x1=X[0]
			x2=X[1]
		else:
			x1=X[:,0]
			x2=X[:,1]
		fval = 100*(x2-x1**2)**2 + (x1-1)**2
		return self.ismax*fval


class beale(functions):
    '''
    beale function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict({'x1':(-4.5,4.5),'x2':(-4.5,4.5)})
        else: self.bounds = bounds
        self.min = [(3, 0.5)]
        self.fmin = 0
        self.ismax=-1
        self.name = 'Beale'

    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]	
        fval = (1.5-x1+x1*x2)**2+(2.25-x1+x1*x2**2)**2+(2.625-x1+x1*x2**3)**2
        return self.ismax*fval   
			


class dropwave(functions):
    '''
    dropwave function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-5.12,5.12)),('x2',(-5.12,5.12))])
        else: self.bounds = bounds
        self.min = [(0, 0)]
        self.fmin = -1
        self.ismax=-1
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'dropwave'

    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        
        fval = - (1+np.cos(12*np.sqrt(x1**2+x2**2))) / (0.5*(x1**2+x2**2)+2) 

        return self.ismax*fval


class cosines(functions):
    '''
    Cosines function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(0,1)),('x2',(0,1))])
        else: self.bounds = bounds
        self.min = [(0.31426205,  0.30249864)]
        self.fmin = -1.59622468
        self.ismax=-1
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Cosines'

    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        #X = reshape(X,self.input_dim)
        #n = X.shape[0]
        
        u = 1.6*x1-0.5
        v = 1.6*x2-0.5
        fval = 1-(u**2 + v**2 - 0.3*np.cos(3*np.pi*u) - 0.3*np.cos(3*np.pi*v) )

        return self.ismax*fval
            
            
            
class goldstein(functions):
    '''
    Goldstein function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = {'x1':(-2,2),'x2':(-2,2)}
        else: self.bounds = bounds
        self.min = [(0,-1)]
        self.fmin = 3
        self.ismax=-1

        self.name = 'Goldstein'


    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
   
        fact1a = (x1 + x2 + 1)**2
        fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2
        fact1 = 1 + fact1a*fact1b
        fact2a = (2*x1 - 3*x2)**2
        fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2
        fact2 = 30 + fact2a*fact2b
        fval = fact1*fact2

        return self.ismax*fval



class sixhumpcamel(functions):
    '''
    Six hump camel function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = OrderedDict([('x1',(-2,2)),('x2',(-1,1))])
        else: self.bounds = bounds
        self.min = [(0.0898,-0.7126),(-0.0898,0.7126)]
        self.fmin = -1.0316
        self.ismax=-1
        
        self.name = 'Six-hump camel'
		
    def func(self,X):
        X=np.asarray(X)
        if len(X.shape)==1:
            x1=X[0]
            x2=X[1]
        else:
            x1=X[:,0]
            x2=X[:,1]
        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1*x2
        term3 = (-4+4*x2**2) * x2**2
        fval = term1 + term2 + term3
        return self.ismax*fval



class mccormick(functions):
    '''
    Mccormick function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1.5,4),(-3,4)]
        else: self.bounds = bounds
        self.min = [(-0.54719,-1.54719)]
        self.fmin = -1.9133
        self.ismax=-1
        self.name = 'Mccormick'

    def func(self,X):
        x1=X[0]
        x2=X[1]
 
      
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2)**2
        term3 = -1.5*x1
        term4 = 2.5*x2
        fval = term1 + term2 + term3 + term4 + 1
        return self.ismax*fval


class powers(functions):
    '''
    Powers function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        if bounds == None: self.bounds = [(-1,1),(-1,1)]
        else: self.bounds = bounds
        self.min = [(0,0)]
        self.fmin = 0
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Sum of Powers'

    def func(self,x):
        x = reshape(x,self.input_dim)
        n = x.shape[0]
        if x.shape[1] != self.input_dim:
            return 'wrong input dimension'
        else:
            x1 = x[:,0]
            x2 = x[:,1]
            fval = abs(x1)**2 + abs(x2)**3
            if self.sd ==0:
                noise = np.zeros(n).reshape(n,1)
            else:
                noise = np.random.normal(0,self.sd,n).reshape(n,1)
            return fval.reshape(n,1) + noise

class eggholder:
    def __init__(self,bounds=None,sd=None):
        self.input_dim = 2
        #self.bounds = {'x1':(-512,512),'x2':(-512,512)}
        self.bounds = [(-512,512),(-512,512)]
        
        self.min = [(512,404.2319)]
        self.fmin = -959.6407
        self.ismax=-1
        if sd==None: self.sd = 0
        else: self.sd=sd
        self.name = 'Egg-holder'

    def func(self,X):
        #x1=X[:,0]
        #x2=X[:,1]
        x1=X[0]
        x2=X[1]
        fval = -(x2+47) * np.sin(np.sqrt(abs(x2+x1/2+47))) + -x1 * np.sin(np.sqrt(abs(x1-(x2+47))))
        
        return self.ismax*fval

class alpine1:
    '''
    Alpine1 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(-10,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd
            
        self.ismax=-1
        self.name='alpine1'

    def func(self,X):
        X = reshape(X,self.input_dim)
        #n = X.shape[0]
        temp=(X*np.sin(X) + 0.1*X)
        if len(temp.shape)<=1:
            fval=np.sum(temp)
        else:
            fval = np.sum(temp,axis=1)
        
        return self.ismax*fval


class alpine2:
    '''
    Alpine2 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(1,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(7.917)]*input_dim
        self.fmin = -2.808**input_dim
        self.ismax=-1
        self.input_dim = input_dim
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd
        self.name='Alpine2'
    def internal_func(self,X):
        fval = np.cumprod(np.sqrt(X))[self.input_dim-1]*np.cumprod(np.sin(X))[self.input_dim-1]  
        return fval

    def func(self,X):
        #X = reshape(X,self.input_dim)
        #n=X.shape[0]
        #n = X.shape[0]
        #fval = np.cumprod(np.sqrt(X),axis=1)[:,self.input_dim-1]*np.cumprod(np.sin(X),axis=1)[:,self.input_dim-1]  
        #fval = np.cumprod(np.sqrt(X))[:,self.input_dim-1]*np.cumprod(np.sin(X))[:,self.input_dim-1]
        fval=[self.ismax*self.internal_func(val) for idx, val in enumerate(X)]

        return fval

class gSobol:
    '''
    gSolbol function
   
    :param a: one-dimensional array containing the coefficients of the function.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,a,bounds=None,sd=None):
        self.a = a
        self.input_dim = len(self.a)

        if bounds == None: 
            self.bounds =[(-4,6)]*self.input_dim
        else: 
            self.bounds = bounds

        if not (self.a>0).all(): return 'Wrong vector of coefficients, they all should be positive'
        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd==None: self.sd = 0
        else: self.sd=sd

        self.ismax=-1
        self.fmin=0# in correct
        self.name='gSobol'

    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n,1)*self.a)/(1+np.ones(n).reshape(n,1)*self.a)
        fval =  np.cumprod(aux,axis=1)[:,self.input_dim-1]

        return self.ismax*fval

#####
class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds == None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='ackley'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        #print X
        #n = X.shape[0]
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))
        
      
        return self.ismax*fval


#####
class hartman_6d:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 6

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax=-1
        self.name='hartman_6d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
			[2329, 4135, 8307, 3736, 1004, 9991],
			[2348, 1451, 3522, 2883, 3047, 6650],
			[4047, 8828, 8732, 5743, 1091, 381]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(6):
                    xj = X[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
				
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -(2.58 + outer) / 1.94;
        
      
        if n==1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
        
        
#####
class hartman_4d:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.32237
        self.ismax=-1
        self.name='hartman_4d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[10, 3, 17, 3.5, 1.7, 8],
             [0.05, 10, 17, 0.1, 8, 14],
             [3, 3.5, 1.7, 10, 17, 8],
             [17, 8, 0.05, 10, 0.1, 14]]
        A=np.asarray(A)
        P = [[1312, 1696, 5569, 124, 8283, 5886],
			[2329, 4135, 8307, 3736, 1004, 9991],
			[2348, 1451, 3522, 2883, 3047, 6650],
			[4047, 8828, 8732, 5743, 1091, 381]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;
        

        fval  =np.zeros((n,1))        
        for idx in range(n):
            X_idx=X[idx,:]
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(4):
                    xj = X_idx[jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                inner = inner + Aij*(xj-Pij)**2
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new
            fval[idx] = (1.1 - outer) / 0.839;
        if n==1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
            
            
            
class hartman_3d:
    '''
    hartman_3d: function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3

        if bounds == None: 
            self.bounds =[(0,1)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = -3.86278
        self.ismax=-1
        self.name='hartman_3d'
        
    def func(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]

        
        alpha = [1.0, 1.2, 3.0, 3.2];
        
        A = [[3.0, 10, 30],
             [0.1, 10, 35],
             [3.0, 10, 30],
             [0.1, 10, 35]]
        A=np.asarray(A)
        P = [[3689, 1170, 2673],
			[4699, 4387, 7470],
			[1091, 8732, 5547],
			[381, 5743, 8828]]



        P=np.asarray(P)
        c=10**(-4)       
        P=np.multiply(P,c)
        outer = 0;

        fval  =np.zeros((n,1))  
        for idx in range(n):
            outer = 0;
            for ii in range(4):
                inner = 0;
                for jj in range(3):
                    xj = X[idx,jj]
                    Aij = A[ii, jj]
                    Pij = P[ii, jj]
                    inner = inner + Aij*(xj-Pij)**2
				
                new = alpha[ii] * np.exp(-inner)
                outer = outer + new

            fval[idx] = -outer;
        
      
        if n==1:
            return self.ismax*(fval[0][0])
        else:
            return self.ismax*(fval)
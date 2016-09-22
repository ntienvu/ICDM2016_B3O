# -*- coding: utf-8 -*-
"""
Created on Thu Mar 03 12:34:13 2016

"""

# define Gaussian Process class

from __future__ import division
import numpy as np
from acquisition_functions import AcquisitionFunction, unique_rows

#from sklearn.gaussian_process import GaussianProcess
#from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.metrics.pairwise import pairwise_distances

class PradaGaussianProcess(object):
    
    def __init__ (self,param):
        # init the model
    
        # theta for RBF kernel exp( -theta* ||x-y||)
        self.theta=param['theta']
        self.nGP=0
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
        self.KK_x_x_inv=[]
    
        self.X=[]
        self.Y=[]
    def fit(self,x,y):
        """
        Fit Gaussian Process model

        Input Parameters
        ----------
        x: the observed points 
        y: the outcome y=f(x)
        
        """ 
        self.X=x
        self.Y=y
        
        ur = unique_rows(self.X)
        x=x[ur]
        y=y[ur]
        
        Euc_dist=euclidean_distances(x,x)
        self.KK_x_x=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        self.KK_x_x_inv=np.linalg.pinv(self.KK_x_x)
        
    
    def fit_incremental(self,newX,newY):
        """
        fit Gaussian Process incrementally using newX and newY
        
        Input Parameters
        ----------
        newX: the new observed points
        newY: the new testing points newY=f(newX)
        
        """         
        
        nNew=len(newY)
        newX=np.reshape(newX,(nNew,-1))
        newY=np.reshape(newY,(nNew,-1))
        #K_xtest_xtrain
        Euc_dist=euclidean_distances(self.X,newX)
        KK_x=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        
        delta_star=np.dot(self.KK_x_x_inv,KK_x)
        sigma=np.identity(nNew)-np.dot(KK_x.T,delta_star)
        inv_sigma=np.linalg.pinv(sigma)
        #sigma=np.diag(sigma)

        temp=np.dot(delta_star,inv_sigma)
        TopLeft=self.KK_x_x_inv+np.dot(temp,delta_star.T)
        #TopLeft=self.KK_x_x_inv+np.dot(delta_star,delta_star.T)/sigma
        #TopRight=-np.divide(delta_star,sigma)
        TopRight=-np.dot(delta_star,np.linalg.pinv(sigma))
        #BottomLeft=-np.divide(delta_star.T,sigma)
        BottomLeft=-np.dot(inv_sigma,delta_star.T)
        #BottomRight=np.divide(np.identity(nNew),sigma)
        BottomRight=np.dot(np.identity(nNew),inv_sigma)

        
        new_K_inv=np.vstack((TopLeft,BottomLeft))
        temp=np.vstack((TopRight,BottomRight))
        self.KK_x_x_inv=np.hstack((new_K_inv,temp))
        self.X=np.vstack((self.X,newX))
        

        self.Y=np.hstack((self.Y.ravel(),newY.ravel()))

    def compute_incremental_var(self,X,newX):
        """
        Compute covariance matrix incrementall for BUCB (KK_x_x_inv_bucb)
        
        Input Parameters
        ----------
        X: the observed points 
        newX: the new point
        
        Returns
        -------
        KK_x_x_inv_bucb: the covariance matrix will be incremented one row and one column
        """   
        
        if len(newX.shape)==1: # 1d
            newX=newX.reshape((-1,newX.shape[0]))
            
        nNew=np.shape(newX)[0]
        #K_xtest_xtrain
        Euc_dist=euclidean_distances(X,newX)
        KK_x=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        
        delta_star=np.dot(self.KK_x_x_inv_bucb,KK_x)
        sigma=np.identity(nNew)-np.dot(KK_x.T,delta_star)
        inv_sigma=np.linalg.pinv(sigma)


        temp=np.dot(delta_star,inv_sigma)
        TopLeft=self.KK_x_x_inv_bucb+np.dot(temp,delta_star.T)
        TopRight=-np.dot(delta_star,np.linalg.pinv(sigma))
        BottomLeft=-np.dot(inv_sigma,delta_star.T)
        BottomRight=np.dot(np.identity(nNew),inv_sigma)

        
        new_K_inv=np.vstack((TopLeft,BottomLeft))
        temp=np.vstack((TopRight,BottomRight))
        self.KK_x_x_inv_bucb=np.hstack((new_K_inv,temp))
                
                
        #Euc_dist=euclidean_distances(newX,newX)
        #KK_xTest_xTest=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        #temp=np.dot(KK_x.T,self.KK_x_x_inv_bucb)
        #var=KK_xTest_xTest-np.dot(temp,KK_x)        
        #return np.diag(var)  

    def compute_var(self,X,xTest):
        """
        compute variance given X and xTest
        
        Input Parameters
        ----------
        X: the observed points
        xTest: the testing points 
        
        Returns
        -------
        diag(var)
        """ 
            
        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        Euc_dist=euclidean_distances(xTest,X)
        KK_xTest_xTrain=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)
            
        return np.diag(var)  
        
    def predict_bucb(self,xTest,eval_MSE):
        """
        compute predictive mean and variance for BUCB        
        
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """
    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
            
        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        Euc_dist=euclidean_distances(xTest,self.X)
        KK_xTest_xTrain=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        #mean=np.dot(temp.T,self.Y)
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,self.Y)
        
        Euc_dist=euclidean_distances(xTest,self.X_bucb)
        KK_xTest_xTrain_bucb=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        temp=np.dot(KK_xTest_xTrain_bucb,self.KK_x_x_inv_bucb)
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain_bucb.T)
            
        return mean.ravel(),np.diag(var)

        
    def predict(self,xTest,eval_MSE):
        """
        compute predictive mean and variance
        Input Parameters
        ----------
        xTest: the testing points 
        
        Returns
        -------
        mean, var
        """    
        if len(xTest.shape)==1: # 1d
            xTest=xTest.reshape((-1,self.X.shape[1]))
        
            
        Euc_dist=euclidean_distances(xTest,xTest)
        KK_xTest_xTest=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        Euc_dist=euclidean_distances(xTest,self.X)
        KK_xTest_xTrain=np.exp(-self.theta*np.square(Euc_dist))+self.noise_delta
        
        #mean=np.dot(temp.T,self.Y)
        temp=np.dot(KK_xTest_xTrain,self.KK_x_x_inv)
        mean=np.dot(temp,self.Y)
        
        var=KK_xTest_xTest-np.dot(temp,KK_xTest_xTrain.T)

            
        return mean.ravel(),np.diag(var)  

        
    def posterior(self,x):
        # compute mean function and covariance function
        return self.predict(self,x)
        
    def predictive_gradient(self,X,Y,xnew):
        """
        Compute predictive gradient to estimat Lipschit constant
        
        Input Parameters
        ----------
        X: The observed points
        Y: The evaluated outcome Y=f(X)
        xnew: the new points 
        
        Returns
        -------
        mean_derivative (\delta_mu)
        """
        
        # available for RBF kernel
        # x1 - xnew
        #x1_xnew=[x-xnew for x in X]
        #x1_xnew=np.asarray(x1_xnew)
        #x1_xnew=pairwise_distances(xnew,X,metric='manhattan')
        
        # compute gradient for each dimension
        if len(xnew.shape)==1:
            ndim=len(xnew)
            NN=len(X)
        else:
            ndim=xnew.shape[1]
            NN=X.shape[0]
        
        Y=np.reshape(Y,NN,-1)
        #X=np.reshape(X,NN,-1)
        
        if ndim>1:
            mean_derivative=np.zeros((xnew.size/ndim,ndim))
            for dd in range(ndim):
    
                # check vector or matrix
                if xnew.size==xnew.shape[0] & xnew.shape[0]!=500: # vector
                    temp=np.subtract(X[:,dd], xnew[dd])
                    #temp=np.sum(temp,axis=1)
                else:
                    #temp=[np.sum( np.subtract(X, x_i), axis=1) for x_i in xnew ]
                    temp=[ np.subtract(X[:,dd], x_i) for x_i in xnew[:,dd] ]
                
                x1_xnew=np.asarray(temp)
                
                # ||x1-xnew||^2
                Euc_dist=euclidean_distances(xnew,X)
        
                #out=-self.theta*2*x1_xnew*np.exp(-self.theta*Euc_dist)
                out=-self.theta*2*x1_xnew*np.exp(-self.theta*np.square(Euc_dist))
                
                Euc_dist_X_X=euclidean_distances(X,X)
                KK_x_x=np.exp(-self.theta*np.square(Euc_dist_X_X))+self.noise_delta
                #KK_x_x=self.KK_x_x


                try:
                    temp=np.linalg.solve(KK_x_x,out.T)
                except:
                    temp=np.zeros(self.Y.shape)
                    
                myproduct=np.dot(temp.T,Y)    
                try:
                    mean_derivative[:,dd]=np.atleast_2d(np.dot(temp.T,Y))
                except:
                    #mean_derivative[:,dd]=np.atleast_1d(np.dot(temp.T,Y))
                    mean_derivative[:,dd]=np.reshape(myproduct,-1,1)
                
                
        else:
                # check vector or matrix
                if xnew.size==xnew.shape[0] & xnew.shape[0]!=500: # vector
                    temp=np.subtract(X[:,0], xnew)
                else:
                    temp=[ np.subtract(X[:,0], x_i) for x_i in xnew[:,0] ]
                
                x1_xnew=np.asarray(temp)
                
                # ||x1-xnew||^2
                Euc_dist=euclidean_distances(xnew,X)
        
                #out=-self.theta*2*x1_xnew*np.exp(-self.theta*Euc_dist)
                out=-self.theta*2*x1_xnew*np.exp(-self.theta*np.square(Euc_dist))
                
                Euc_dist_X_X=euclidean_distances(X,X)
                KK_x_x=np.exp(-self.theta*np.square(Euc_dist_X_X))+self.noise_delta
                
                try:
                    temp=np.linalg.solve(KK_x_x,out.T)
                except:
                    temp=np.zeros(self.Y.shape)    
                #temp=np.atleast_2d(temp)
                #Y=np.atleast_2d(Y)
                #print temp.shape
                #print Y.shape
                mean_derivative=np.dot(temp.T,Y.T)
        
        return mean_derivative
        
# ======================================================================================    
# ======================================================================================
# ======================================================================================
            
            
class PradaMultipleGaussianProcess(object):
    
    def __init__ (self,**param):
        # init the model
    
        # theta for RBF kernel exp( -theta* ||x-y||)
        self.theta=param['theta']
        self.nGP=len(param['theta'])
        
        # noise delta is for GP version with noise
        self.noise_delta=param['noise_delta']
        
        self.KK_x_x=[]
    
        self.X=[]
        self.Y=[]
        
    def fit(self,x,y):
        # fit the model, given the observation x and y        
        self.X=x
        self.Y=y
        Euc_dist=euclidean_distances(x,x)
        
        self.KK_x_x=[]
        for idx in range(self.nGP):
            temp=np.exp(-self.theta[idx]*np.square(Euc_dist))+self.noise_delta
            self.KK_x_x.append(temp)
            
    
    def predict(self,xTest,eval_MSE):
        # predict y value given x and model  
        Euc_dist_test=euclidean_distances(xTest,xTest)
        Euc_dist_train_test=euclidean_distances(self.X,xTest)

        KK_xTest_xTest=[]
        KK_xTest_xTrain=[]
        
        mean=[]
        var=[]
        for idx in range(self.nGP):
            temp=np.exp(-self.theta[idx]*np.square(Euc_dist_test))+self.noise_delta
            KK_xTest_xTest.append(temp)
        
            temp2=np.exp(-self.theta[idx]*np.square(Euc_dist_train_test))+self.noise_delta
            KK_xTest_xTrain.append(temp2)
            
            temp=np.linalg.solve(self.KK_x_x[idx],KK_xTest_xTrain[idx])
                
            temp_mean=np.dot(temp.T,self.Y)
            mean.append(temp_mean)
            
            temp_var=(KK_xTest_xTest[idx]-np.dot(temp.T,KK_xTest_xTrain[idx]))
            var.append(np.diag(temp_var))
            
        if len(xTest)==1000:
            return mean,var
        else:
            return np.asarray(mean),np.asarray(var)

        
    def posterior(self,x):
        # compute mean function and covariance function
        return predict(self,x)
    
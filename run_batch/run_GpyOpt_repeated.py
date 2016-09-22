# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:18:00 2016

"""
import pickle

import sys
sys.path.insert(0,'..')
from prada_bayes_opt import functions
import itertools
import GPyOpt
import numpy as np
from prada_bayes_opt import auxiliary_functions
from numpy.random import seed
seed(12345)

#myfunction2=functions.gSobol(a=np.array([1,1,1,1,1]))
#myfunction2=functions.alpine2(input_dim=5)
#myfunction2=functions.forrester()
myfunction2=functions.dropwave()

pbound=myfunction2.bounds

#convert mybound from dict to array
if isinstance(pbound,dict):
    # Get the name of the parameters
    mybounds = []
    for key in pbound.keys():
        mybounds.append(pbound[key])
    mybounds = np.asarray(mybounds)
else:
    mybounds=np.asarray(pbound)
            
yoptimal=myfunction2.fmin*myfunction2.ismax*-1
    
# --- Objective function
#myfunction  = GPyOpt.fmodels.experimentsNd.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))             
#myfunction  = GPyOpt.fmodels.experimentsNd.gSobol(a=np.array([1,1,1,1,1]))             
#myfunction  = GPyOpt.fmodels.experimentsNd.alpine2(input_dim=5)             
#myfunction  = GPyOpt.fmodels.experimentsNd.alpine2(input_dim=10)             
#myfunction  = GPyOpt.fmodels.experimentsNd.hartman_3d()
#myfunction  = GPyOpt.fmodels.experimentsNd.hartman_6d()
#myfunction  = GPyOpt.fmodels.experiments1d.forrester()
myfunction  = GPyOpt.fmodels.experiments2d.dropwave()


nRepeat=20

GAP=[0]*nRepeat
ybest=[0]*nRepeat
Regret=[0]*nRepeat
BatchSz=[0]*nRepeat
MyOptTime=[0]*nRepeat
MyTime=[0]*nRepeat

# loop over all acquisition function type                       
acq_type_list={'LCB','EI'}         
# loop over all batch BO approaches
mybatch_type_list={'random','predictive','lp'}

print "This will take time......................................................"

for idx, (acq_type,mybatch_type) in enumerate(itertools.product(acq_type_list,mybatch_type_list)):
    
    for ii in range(nRepeat):
        # --- Problem definition and optimization
        bo_gpy = GPyOpt.methods.BayesianOptimization(f=myfunction.f,
                                                bounds = mybounds,acquisition = acq_type,
                                                acquisition_par = 0,normalize = True)
                                                
        bo_gpy.dim=myfunction.input_dim  
        
        
        GAP[ii],ybest[ii],Regret[ii],MyTime[ii],MyOptTime[ii]=auxiliary_functions.run_experiment_batch_GPyOpt(bo_gpy,yoptimal,batch_type=mybatch_type,
                                                                n_init=bo_gpy.dim*3,NN=10*bo_gpy.dim)
        BatchSz[ii]=len(bo_gpy.Y)
        
        #if np.mod(ii,5)==0:
            #now = time.strftime("%c")
            #print 'ID={:d} {:s} GAP={:.2f} Min Best={:.3f}'.format(ii,now,GAP[ii],ybest[ii].min())
    
    Score={}
    Score["GAP"]=GAP
    Score["ybest"]=ybest
    Score["Regret"]=Regret
    Score["MyTime"]=MyTime
    Score["BatchSz"]=BatchSz
    Score["MyOptTime"]=MyOptTime
    auxiliary_functions.print_result(myfunction2,Score,mybatch_type,acq_type,toolbox='GPyOpt')        
    
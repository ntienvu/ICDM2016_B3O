# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 21:37:03 2016

"""


import numpy as np
import time
import pickle
import os

from prada_bayes_opt import *

        
def run_experiment(bo,gp_params,yoptimal,n_init=3,NN=10):
    # create an empty object for BO
    

    start_time = time.time()
    bo.init(n_init_points=n_init)
    
    # number of recommended parameters
    #NN=10*bo.dim
    for index in range(0,NN-1):
        bo.maximize(gp_params,init_points=0,kappa=2)

    
    y_init=np.max(bo.Y[0:n_init])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    
    if GAP<0.00001:
        GAP=0
    fxoptimal=bo.Y
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time
    
def run_experiment_batch(bo,gp_params,yoptimal,batch_type='gmm',B=3,n_init=3,NN=10):
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
 
    start_time = time.time()

    # Initialize x, y and find current y_max
    bo.init(n_init_points=n_init)
        
    # number of recommended parameters

    for index in range(0,NN):# because it has already runned another maximize step above

        if batch_type=="b3o":
            bo.maximize_batch_B3O(gp_params,kappa=2)
        if batch_type=="bucb":
            bo.maximize_batch_BUCB(gp_params,B=B,kappa=2) 
            
    # evaluation
    y_init=np.max(bo.Y[0:B])
    GAP=(y_init-bo.Y.max())*1.0/(y_init-yoptimal)
    
    
    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo.Y)]
    #Regret=[np.abs(val-yoptimal) for idx,val in enumerate(current_best_y)] 
    
    elapsed_time = time.time() - start_time

    # we also store the optimization time in bo.opt_time
    return GAP, Regret,elapsed_time
    
def run_experiment_batch_GPyOpt(bo_gpy,yoptimal,batch_type='lp',B=3,n_init=3,NN=10):
    #  this function will run the GPyOpt toolbox for Constant Liar (Pred), Local Penalization and Random methods

    start_time = time.time()
    
    myinit_points=n_init
    
    # --- Run the optimization    
    if bo_gpy.dim<5:                                                          
        B = 3                                                          
    else:
        B = bo_gpy.dim                                                          

    # --- Run the optimization                                              
    bo_gpy.run_optimization(max_iter=NN,acqu_optimize_method = 'random',
                            n_inbatch = B,batch_method=batch_type,
                            acqu_optimize_restarts = 5,eps = 1e-6)                            
       
    # evaluation
    y_init=np.max(bo_gpy.Y[0:myinit_points])
    GAP=(y_init-bo_gpy.Y.min())*1.0/(y_init-yoptimal)
    

    Regret=[np.abs(val-yoptimal) for idx,val in enumerate(bo_gpy.Y)]
    #Regret=[np.abs(val-yoptimal) for idx,val in enumerate(current_best_y)]
  
    fxoptimal=bo_gpy.Y
    
    elapsed_time = time.time() - start_time

    return GAP, fxoptimal, Regret,elapsed_time,bo_gpy.opt_time
    
    
def print_result(myfunction,Score,mybatch_type,acq_type,toolbox='GPyOpt'):
    Regret=Score["Regret"]
    ybest=Score["ybest"]
    GAP=Score["GAP"]
    MyTime=Score["MyTime"]
    

    print myfunction.name

    AveRegret=[np.mean(val) for idx,val in enumerate(Regret)]
    StdRegret=[np.std(val) for idx,val in enumerate(Regret)]
    
    if toolbox=='GPyOpt':
        MaxFx=[val.min() for idx,val in enumerate(ybest)]
    else:
        MaxFx=[val.max() for idx,val in enumerate(ybest)]
        
    print '[{:s} {:s} {:s}] GAP={:.3f}({:.2f}) AvgRegret={:.3f}({:.2f}) ElapseTime={:.3f}({:.2f})'\
                .format(toolbox,mybatch_type,acq_type,np.mean(GAP),np.std(GAP),np.mean(AveRegret),\
                np.std(StdRegret),np.mean(MyTime),np.std(MyTime))
    
    if toolbox=='GPyOpt':
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.2f})'.format(-1*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.2f})'.format(np.mean(MaxFx),np.std(MaxFx))
    else:            
        if myfunction.ismax==1:
            print 'MaxBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))    
        else:
            print 'MinBest={:.4f}({:.2f})'.format(myfunction.ismax*np.mean(MaxFx),np.std(MaxFx))
            
    if 'BatchSz' in Score:
        BatchSz=Score["BatchSz"]
        if toolbox=='GPyOpt':
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(BatchSz),np.std(BatchSz))
        else:
            SumBatch=np.sum(BatchSz,axis=1)
            print 'BatchSz={:.3f}({:.2f})'.format(np.mean(SumBatch),np.std(SumBatch))
    
    if 'MyOptTime' in Score:
        MyOptTime=Score["MyOptTime"]
        if toolbox=='GPyOpt':
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(MyOptTime),np.std(MyOptTime))
        else:
            SumOptTime=np.sum(MyOptTime,axis=1)
            print 'OptTime={:.1f}({:.1f})'.format(np.mean(SumOptTime),np.std(SumOptTime))
            

        
    out_dir="P:\\01.Private\\03.Research\\05.BayesianOptimization\\PradaBayesianOptimization\\pickle_storage"
    strFile="{:s}_{:d}_{:s}_{:s}.pickle".format(myfunction.name,myfunction.input_dim,mybatch_type,acq_type)
    path=os.path.join(out_dir,strFile)
    with open(path, 'w') as f:
        if 'BatchSz' in Score:
            pickle.dump([ybest, Regret, MyTime,BatchSz], f)
        else:
            pickle.dump([ybest, Regret, MyTime], f)
            
def yBest_Iteration(YY,BatchSzArray,IsPradaBO=0):

    temp=YY[0:BatchSzArray[0]+1].max()
    start_point=0
    for idx,bz in enumerate(BatchSzArray):
        if idx==len(BatchSzArray)-1:
            break
        bz=np.int(bz)
        
        if IsPradaBO==1:
            temp=np.vstack((temp,YY[0:start_point+bz+1].max()))
        else:
            temp=np.vstack((temp,YY[0:start_point+bz+1].min()))
        start_point=start_point+bz
        
    temp=np.array(temp)
    
    if IsPradaBO==1:
        myYbest=[temp[:idx+1].max()*-1 for idx,val in enumerate(temp)]
    else:
        myYbest=[temp[:idx+1].min() for idx,val in enumerate(temp)]
    return myYbest
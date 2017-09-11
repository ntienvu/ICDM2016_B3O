import sys
sys.path.insert(0,'..')

from prada_bayes_opt import PradaBayOptBatch
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
#from my_plot_gp import plot_gp
#from my_plot_gp import plot_gp_batch

from prada_bayes_opt import auxiliary_functions

from prada_bayes_opt import functions
from prada_bayes_opt import real_experiment_function
import pickle
import random
import time

import warnings
warnings.filterwarnings("ignore")
#%matplotlib inline.

random.seed('6789')

#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))    
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))
#myfunction=functions.hartman_3d()
#myfunction=functions.hartman_6d()
#myfunction=functions.beale()
#myfunction=functions.forrester()
#myfunction=functions.alpine2(input_dim=5)
#myfunction=functions.alpine2(input_dim=10)
myfunction=functions.dropwave()



func=myfunction.func


# define a bound for the parameter

mybound=myfunction.bounds

yoptimal=myfunction.fmin*myfunction.ismax

gp_params = {'theta':1*myfunction.input_dim,'noise_delta':0.000001}

nRepeat=3

GAP=[0]*nRepeat
ybest=[0]*nRepeat
Regret=[0]*nRepeat
BatchSz=[0]*nRepeat
MinBatchSz=[0]*nRepeat
MaxBatchSz=[0]*nRepeat
MyTime=[0]*nRepeat
MyOptTime=[0]*nRepeat


acq_type='ucb'
mybatch_type='b3o'
for ii in range(nRepeat):
    bo=PradaBayOptBatch(func,mybound,acq=acq_type,opt='scipy')
    GAP[ii],Regret[ii],MyTime[ii]=auxiliary_functions.run_experiment_batch(bo,gp_params,yoptimal,batch_type=mybatch_type,n_init=3,NN=10*myfunction.input_dim)
    MyOptTime[ii]=bo.opt_time
    ybest[ii]=bo.Y
    BatchSz[ii]=bo.NumPoints
    MinBatchSz[ii]=np.min(bo.NumPoints)
    MaxBatchSz[ii]=np.max(bo.NumPoints)
    if np.mod(ii,1)==0:
        now = time.strftime("%c")

        if myfunction.ismax==1:
            print 'ID={:d} {:s} GAP={:.2f} MaxBest={:.4f} Elapse={:.1f}'.format(ii,now,GAP[ii],myfunction.ismax*ybest[ii].max(),MyTime[ii])
        else:
            print 'ID={:d} {:s} GAP={:.2f} MinBest={:.4f} Elapse={:.1f}'.format(ii,now,GAP[ii],myfunction.ismax*ybest[ii].max(),MyTime[ii])
 

Score={}
Score["GAP"]=GAP
Score["ybest"]=ybest
Score["Regret"]=Regret
Score["MyTime"]=MyTime
Score["MyOptTime"]=MyOptTime
Score["BatchSz"]=BatchSz
print_result(myfunction,Score,mybatch_type,acq_type,toolbox='BatchBO')        

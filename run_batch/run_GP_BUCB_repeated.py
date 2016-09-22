# this script runs the baselines GP-BUCB for comparison

import sys
sys.path.insert(0,'..')
from prada_bayes_opt import PradaBayOptBatch
import numpy as np
from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt import functions
from prada_bayes_opt import real_experiment_function
import time
import warnings
warnings.filterwarnings("ignore")

    
#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))
#myfunction=functions.alpine2(input_dim=10)
#myfunction=real_experiment_function.DeepLearning_MLP_MNIST()
#myfunction=functions.beale()
myfunction=functions.dropwave()
#myfunction=functions.forrester()
#myfunction=real_experiment_function.SVR_function()

#myfunction=functions.alpine2(input_dim=5)

func=myfunction.func

yoptimal=myfunction.fmin*myfunction.ismax

gp_params = {'theta':0.1*myfunction.input_dim,'noise_delta':0.1}

nRepeat=10

GAP=[0]*nRepeat
ybest=[0]*nRepeat
Regret=[0]*nRepeat
BatchSz=[0]*nRepeat
MinBatchSz=[0]*nRepeat
MaxBatchSz=[0]*nRepeat
MyTime=[0]*nRepeat
MyOptTime=[0]*nRepeat


acq_type='bucb'
mybatch_type='bucb'

for ii in range(nRepeat):
    bo=PradaBayOptBatch(func,myfunction.bounds,acq=acq_type,opt='scipy')
    GAP[ii],Regret[ii],MyTime[ii]=auxiliary_functions.run_experiment_batch(bo,gp_params,yoptimal,
                                    batch_type=mybatch_type,B=3,n_init=3*bo.dim,NN=10)
    MyOptTime[ii]=bo.opt_time
    ybest[ii]=bo.Y
    BatchSz[ii]=bo.NumPoints
    MinBatchSz[ii]=np.min(bo.NumPoints)
    MaxBatchSz[ii]=np.max(bo.NumPoints)
    if np.mod(ii,1)==0:
        now = time.strftime("%c")

        if myfunction.ismax==1:
            print 'ID={:d} {:s} GAP={:.2f} MaxBest={:.3f} Elapse={:.1f}'.format(ii,now,GAP[ii],myfunction.ismax*ybest[ii].max(),MyTime[ii])
        else:
            print 'ID={:d} {:s} GAP={:.2f} MinBest={:.3f} Elapse={:.1f}'.format(ii,now,GAP[ii],myfunction.ismax*ybest[ii].max(),MyTime[ii])
 

Score={}
Score["GAP"]=GAP
Score["ybest"]=ybest
Score["Regret"]=Regret
Score["MyTime"]=MyTime
Score["MyOptTime"]=MyOptTime
Score["BatchSz"]=BatchSz
auxiliary_functions.print_result(myfunction,Score,mybatch_type,acq_type,toolbox='BatchBO')        
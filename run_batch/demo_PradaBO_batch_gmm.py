import sys
sys.path.insert(0,'..')

from prada_bayes_opt import PradaBayOptBatch
from prada_bayes_opt import functions
from prada_bayes_opt import auxiliary_functions
import numpy as np
import matplotlib.pyplot as plt
from auxiliary_functions import *
from prada_bayes_opt import *

import random
import warnings
warnings.filterwarnings("ignore")

random.seed('6789')

# please select one of the functions / exepriments below

#myfunction=functions.forrester()       #1D
#myfunction=functions.branin()          #2D
#myfunction=functions.dropwave()        #2D

myfunction=functions.hartman_3d()      #3D

#myfunction=real_experiment_function.SVR()      #4D Real Experiment with SVR

#myfunction=functions.gSobol(a=np.array([1,1,1,1,1]))       #5D
#myfunction=functions.alpine2(input_dim=5)                   #5D

#myfunction=functions.hartman_6d()                          #6D

#myfunction=real_experiment_function.DeepLearning_MLP_MNIST()       #7D Real Experiment with SV

#myfunction=functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1]))     #10D
#myfunction=functions.alpine2(input_dim=10)                         #10D

print "======================================================================="
print "You are selecting function {:s} D={:d}".format(myfunction.name,myfunction.input_dim)

if myfunction.input_dim<=2:    
    visualization.plot_original_function(myfunction)

# create an empty object for BO
bo=PradaBayOptBatch(f=myfunction.func, pbounds=myfunction.bounds, acq='ucb',opt='scipy')

# parameter for Gaussian Process
gp_params = {'theta':0.1*bo.dim,'noise_delta':0.1}

# init Bayesian Optimization
print "======================================================================="
print "Start Initialization"
bo.init(n_init_points=3*bo.dim)

print "======================================================================="
print "\nRunning Budgeted Batch Bayesian Optimization"


# number of iteration
TT=5*myfunction.input_dim
for index in range(0,TT):
    bo.maximize_batch_B3O(gp_params,kappa=2,IsPlot=1)
    sys.stdout.write("\nIter={:d} Optimization Time={:.2f} sec ".format(index,bo.opt_time[index]))


print "======================================================================="
print "\nB3O #TotalPoints={:.0f} Best-found-value={:.3f}".format(np.sum(bo.NumPoints),bo.Y.max())
idxMax=np.argmax(bo.Y)
print "X_optimal "
print bo.X_original[idxMax]


# plot the best-found-value
my_yBest=auxiliary_functions.yBest_Iteration(bo.Y,bo.NumPoints,IsPradaBO=1)

plt.plot(range(0,TT+1),my_yBest,linewidth=2,color='r',linestyle='-', marker='s',label='B3O')

plt.ylabel('Best-found-value',fontdict={'size':18})
plt.xlabel('Iteration',fontdict={'size':18})
plt.legend(loc=1,prop={'size':18})
#plt.ylim([np.min(my_yBest)*0.7,np.max(my_yBest)*1.2])
strTitle="{:s} D={:d}".format(myfunction.name,myfunction.input_dim)
plt.title(strTitle,fontdict={'size':20})

# plot the batch size per iteration
fig=plt.figure(figsize=(9, 4.5))
strNTotal="B3O (N={:.0f})".format(np.sum(bo.NumPoints))
plt.plot(range(1,TT+1),bo.NumPoints[1:],linewidth=2,color='r',linestyle='-',marker='s', label=strNTotal)
plt.ylabel('# BatchSize per Iter',fontdict={'size':18})
plt.xlabel('Iteration',fontdict={'size':18})
plt.legend(loc=1,prop={'size':18})
plt.ylim([np.min(bo.NumPoints[1:])-1,np.max(bo.NumPoints[1:])+1])
plt.title(strTitle,fontdict={'size':20})

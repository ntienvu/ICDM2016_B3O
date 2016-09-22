"""
Created on Sat Feb 27 23:22:32 2016

"""

from __future__ import division
import numpy as np
#from scipy.stats import norm
#import matplotlib as plt
import matplotlib.pyplot as plt
from matplotlib import gridspec


        
#class Visualization(object):
    
    #def __init__(self,bo):
       #self.plot_gp=0     
       #self.posterior=0
       #self.myBo=bo
       
        
def plot_bo(bo):
    if bo.dim==1:
        plot_bo_1d(bo)
    if bo.dim==2:
        plot_bo_2d(bo)
    
def plot_histogram(bo,samples):
    if bo.dim==1:
        plot_histogram_1d(bo,samples)
    if bo.dim==2:
        plot_histogram_2d(bo,samples)

def plot_mixturemodel(g,bo,samples):
    if bo.dim==1:
        plot_mixturemodel_1d(g,bo,samples)
    if bo.dim==2:
        plot_mixturemodel_2d(g,bo,samples)

def plot_mixturemodel_1d(g,bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]

    x_plot = np.linspace(np.min(samples), np.max(samples), len(samples))
    x_plot = np.reshape(x_plot,(len(samples),-1))
    y_plot = g.score_samples(x_plot)[0]
    
    x_plot_ori = np.linspace(np.min(samples_original), np.max(samples_original), len(samples_original))
    x_plot_ori=np.reshape(x_plot_ori,(len(samples_original),-1))
    
    
    fig=plt.figure(figsize=(8, 3))

    plt.plot(x_plot_ori, np.exp(y_plot), color='red')
    plt.xlim(bo.bounds[0,0],bo.bounds[0,1])
    plt.xlabel("X",fontdict={'size':16})
    plt.ylabel("f(X)",fontdict={'size':16})
    plt.title("IGMM Approximation",fontsize=16)
        
def plot_mixturemodel_2d(dpgmm,bo,samples):
    
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    dpgmm_means_original=dpgmm.truncated_means_*bo.max_min_gap+bo.bounds[:,0]

    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myGmm=fig.add_subplot(1,1,1)    



    x1 = np.linspace(bo.scalebounds[0,0],bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0],bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    x_plot=np.c_[x1g.flatten(), x2g.flatten()]
    
    y_plot2 = dpgmm.score_samples(x_plot)[0]
    y_plot2=np.exp(y_plot2)
    #y_label=dpgmm.predict(x_plot)[0]
    
    x1_ori = np.linspace(bo.bounds[0,0],bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0],bo.bounds[1,1], 100)
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)

    CS_acq=myGmm.contourf(x1g_ori,x2g_ori,y_plot2.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    myGmm.scatter(dpgmm_means_original[:,0],dpgmm_means_original[:,1], marker='*',label=u'Estimated Peaks by IGMM', s=100,color='green')    


    myGmm.set_title('IGMM Approximation',fontsize=16)
    myGmm.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    myGmm.set_ylim(bo.bounds[1,0],bo.bounds[1,1])
    myGmm.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


def plot_histogram_2d(bo,samples):
    
    # convert samples from 0-1 to original scale
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]
    
    #fig=plt.figure(figsize=(12, 5))
    fig=plt.figure()
    myhist=fig.add_subplot(1,1,1)
    
    myhist.set_title("Histogram of Samples under Acq Func",fontsize=16)
    
    #xedges = np.linspace(myfunction.bounds['x1'][0], myfunction.bounds['x1'][1], 10)
    #yedges = np.linspace(myfunction.bounds['x2'][0], myfunction.bounds['x2'][1], 10)
    
    xedges = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 10)
    yedges = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 10)

    H, xedges, yedges = np.histogram2d(samples_original[:,0], samples_original[:,1], bins=50)   
    
    #data = [go.Histogram2d(x=vu[:,1],y=vu[:,0])]
    #plot_url = py.plot(data, filename='2d-histogram')

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)
     
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
     
    # Plot 2D histogram using pcolor
    myhist.pcolormesh(xedges,yedges,Hmasked)
    myhist.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    myhist.set_ylim(bo.bounds[1,0], bo.bounds[1,1])

def plot_histogram_1d(bo,samples):
    samples_original=samples*bo.max_min_gap+bo.bounds[:,0]


    fig=plt.figure(figsize=(8, 3))
    fig.suptitle("Histogram",fontsize=16)
    myplot=fig.add_subplot(111)
    myplot.hist(samples_original,50)
    myplot.set_xlim(bo.bounds[0,0],bo.bounds[0,1])
    
    myplot.set_xlabel("Value",fontsize=16)
    myplot.set_ylabel("Frequency",fontsize=16)
        
    
def plot_bo_1d(bo):
    func=bo.f
    x_original = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 1000)
    x = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 1000)
    y = func(x_original)
    
    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = bo.posterior(x)
    axis.plot(x_original, y, linewidth=3, label='Target')
    axis.plot(bo.X_original.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x_original, mu, '--', color='k', label='Prediction')
    
    #samples*bo.max_min_gap+bo.bounds[:,0]
    
    temp=np.concatenate([x, x[::-1]])
    temp_xaxis=temp*bo.max_min_gap+bo.bounds[:,0]
    
    temp_yaxis=np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]])
    
    axis.fill(temp_xaxis, temp_yaxis,alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x_original), np.max(x_original)))
    #axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    
    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, np.max(bo.Y))
    acq.plot(x_original, utility, label='Utility Function', color='purple')
    acq.plot(x_original[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    acq.plot(bo.X_original[-1][0], np.max(utility), 'v', markersize=15, 
         label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((np.min(x_original), np.max(x_original)))
    #acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    acq.set_ylabel('Acq', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_bo_2d(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
  
    fig = plt.figure()
    
    #axis2d = fig.add_subplot(1, 2, 1)
    acq2d = fig.add_subplot(1, 1, 1)
    
    #mu, sigma = bo.posterior(X)
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b')
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g',label='Observations')    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    
    acq2d.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
            
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
def plot_bo_2d_withGPmeans(bo):
    
    x1 = np.linspace(bo.scalebounds[0,0], bo.scalebounds[0,1], 100)
    x2 = np.linspace(bo.scalebounds[1,0], bo.scalebounds[1,1], 100)
    
    x1g,x2g=np.meshgrid(x1,x2)
    
    X=np.c_[x1g.flatten(), x2g.flatten()]
    
    x1_ori = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    x2_ori = np.linspace(bo.bounds[1,0], bo.bounds[1,1], 100)
    
    x1g_ori,x2g_ori=np.meshgrid(x1_ori,x2_ori)
    
    X_ori=np.c_[x1g_ori.flatten(), x2g_ori.flatten()]
    
    #fig.suptitle('Gaussian Process and Utility Function After {} Points'.format(len(bo.X)), fontdict={'size':18})
    
    fig = plt.figure(figsize=(12, 5))
    
    #axis3d = fig.add_subplot(1, 2, 1, projection='3d')
    axis2d = fig.add_subplot(1, 2, 1)
    #acq3d = fig.add_subplot(2, 2, 3, projection='3d')
    acq2d = fig.add_subplot(1, 2, 2)
    
    mu, sigma = bo.posterior(X)
    #axis.plot(x, y, linewidth=3, label='Target')
    #axis3d.plot_surface(x1g,x1g,mu.reshape(x1g.shape))
    #axis3d.scatter(bo.X[:,0],bo.X[:,1], bo.Y,zdir='z',  label=u'Observations', color='r')    

    
    CS=axis2d.contourf(x1g_ori,x2g_ori,mu.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin='lower',hold='on')
    
    axis2d.scatter(bo.X_original[:,0],bo.X_original[:,1], label=u'Observations', color='r')    
    axis2d.set_title('Gaussian Process Mean',fontsize=16)
    axis2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    axis2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
    #plt.colorbar(ax=axis2d)

    #axis.plot(x, mu, '--', color='k', label='Prediction')
    
    
    #axis.set_xlim((np.min(x), np.max(x)))
    #axis.set_ylim((None, None))
    #axis.set_ylabel('f(x)', fontdict={'size':16})
    #axis.set_xlabel('x', fontdict={'size':16})
    
    # plot the acquisition function

    utility = bo.acq_func.acq_kind(X, bo.gp, np.max(bo.Y))
    #acq3d.plot_surface(x1g,x1g,utility.reshape(x1g.shape))
    
    CS_acq=acq2d.contourf(x1g_ori,x2g_ori,utility.reshape(x1g.shape),cmap=plt.cm.bone,origin='lower')
    CS2_acq = plt.contour(CS_acq, levels=CS_acq.levels[::2],colors='r',origin='lower',hold='on')
    
    idxBest=np.argmax(utility)
    
    acq2d.scatter(X_ori[idxBest,0],X_ori[idxBest,1],color='b')
    acq2d.scatter(bo.X_original[:,0],bo.X_original[:,1],color='g')    
    acq2d.set_title('Acquisition Function',fontsize=16)
    acq2d.set_xlim(bo.bounds[0,0], bo.bounds[0,1])
    acq2d.set_ylim(bo.bounds[1,0], bo.bounds[1,1])
             
    #acq.set_xlim((np.min(x), np.max(x)))
    #acq.set_ylim((np.min(utility), 1.1*np.max(utility)))
    #acq.set_ylabel('Acq', fontdict={'size':16})
    #acq.set_xlabel('x', fontdict={'size':16})
    
    #axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    #acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    
def plot_gp_batch(self,x,y):
    
    bo=self.myBo
    n_batch=bo.n_batch
    
    fig=plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = posterior(bo)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    
    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility = bo.acq_func.acq_kind(x.reshape((-1, 1)), bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    
    #selected_x=x[np.argmax(utility)]
    #selected_y=np.max(utility)
    
    selected_x=bo.X[-1-n_batch:]
    selected_y=utility(selected_x)
    
    acq.plot(selected_x, selected_y,'*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)

def plot_original_function(myfunction):
    
    origin = 'lower'

    func=myfunction.func


    if myfunction.input_dim==1:    
        x = np.linspace(myfunction.bounds['x'][0], myfunction.bounds['x'][1], 1000)
        y = func(x)
    
        fig=plt.figure(figsize=(8, 5))
        plt.plot(x, y)
        strTitle="{:s}".format(myfunction.name)

        plt.title(strTitle)
    
    if myfunction.input_dim==2:    
        x1 = np.linspace(myfunction.bounds['x1'][0], myfunction.bounds['x1'][1], 100)
        x2 = np.linspace(myfunction.bounds['x2'][0], myfunction.bounds['x2'][1], 100)
        x1g,x2g=np.meshgrid(x1,x2)
        X_plot=np.c_[x1g.flatten(), x2g.flatten()]
        Y = func(X_plot)
    
        #fig=plt.figure(figsize=(8, 5))
        
        fig = plt.figure(figsize=(12, 5))
        
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax2d = fig.add_subplot(1, 2, 2)
        
        #ax3d. = fig.gca(projection='3d')
        ax3d.plot_surface(x1g,x2g,Y.reshape(x1g.shape))   
        strTitle3D="{:s} 3D Plot".format(myfunction.name)
        #print strTitle
        ax3d.set_title(strTitle3D)
        #plt.plot(x, y)
        CS=ax2d.contourf(x1g,x2g,Y.reshape(x1g.shape),cmap=plt.cm.bone,origin=origin)   
       
        CS2 = plt.contour(CS, levels=CS.levels[::2],colors='r',origin=origin,hold='on')
        plt.colorbar(ax=ax2d)
        strTitle2D="{:s} 3D Plot".format(myfunction.name)

        plt.title(strTitle2D)
        
def plot_gp_batch(bo,x,y):
    
    n_batch=bo.n_batch
    
    fig=plt.figure(figsize=(8, 5))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':18})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma =  bo.posterior(x)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')
    
    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% CI')
    
    axis.set_xlim((np.min(x), np.max(x)))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':16})
    axis.set_xlabel('x', fontdict={'size':16})
    
    utility = bo.util.utility(x.reshape((-1, 1)), bo.gp, bo.Y.max()) # estimate the acquisition function
    acq.plot(x, utility, label='Utility Function', color='purple')
    
    #selected_x=x[np.argmax(utility)]
    #selected_y=np.max(utility)
    
    #selected_x=np.max(utilily)#bo.X[-n_batch:][0]
    #selected_y=bo.util.utility(selected_x.reshape((-1, 1)), bo.gp,bo.Y.max())
    
    #acq.plot(selected_x, selected_y,'*', markersize=15, 
             #label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
             
    #acq.plot(bo.X[-1][0], np.max(utility), 'v', markersize=15,label=u'Previous Selection', markerfacecolor='green', markeredgecolor='k', markeredgewidth=1)
    
    acq.set_xlim((np.min(x), np.max(x)))
    acq.set_ylim((np.min(utility), np.max(utility)))
    acq.set_ylabel('Utility', fontdict={'size':16})
    acq.set_xlabel('x', fontdict={'size':16})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)




# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 11:25:02 2016

"""

import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVR

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
        
class SVR_function:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 3
        
        if bounds == None: 
            self.bounds = OrderedDict([('C',(0.1,1000)),('epsilon',(0.000001,1)),('gamma',(0.00001,5))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='SVR_function'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_SVR(self,X,X_train,y_train,X_test,y_test):
        x1=X[0]
        x2=X[1]
        x3=X[2]
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        #nTest=X_test.shape[0]
    
        #print x1,x2,x3
        # Fit regression model
        svr_model = SVR(kernel='rbf', C=x1, epsilon=x2,gamma=x3)
        y_pred = svr_model.fit(X_train, y_train).predict(X_test)
        
        
        squared_error=y_pred-y_test
        squared_error=np.mean(squared_error**2)
        
        RMSE=np.sqrt(squared_error)
        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
            
        Xdata, ydata = self.get_data("F:\\Data\\regression\\abalone_scale")
        nTrain=np.int(0.7*len(ydata))
        X_train, y_train = Xdata[:nTrain], ydata[:nTrain]
        X_test, y_test = Xdata[nTrain+1:], ydata[nTrain+1:]
        ###############################################################################
        # Generate sample data

        #y_train=np.reshape(y_train,(nTrain,-1))
        #y_test=np.reshape(y_test,(nTest,-1))
        ###############################################################################

        #print len(X.shape)
        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_SVR(X,X_train,y_train,X_test,y_test)
        else:

            RMSE=np.apply_along_axis( self.run_SVR,1,X,X_train,y_train,X_test,y_test)

        #print RMSE    
        return RMSE*self.ismax
        
class AlloyCooking_Profiling:
    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(2*3600,4*3600)),('Time2',(2*3600,4*3600)),('Temp1',(175,225)),('Temp2',(225,275))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.45;
        myxmatrix=0.0006;
        myiSurfen=0.1097149825;
        myfSurfen=0.1656804095;
        myRadsurfenchange=0.0000000041;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        eng.addpath(r'F:\Dropbox\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange)

        temp=np.asarray(strength)
        return temp[0][1]
        
    def func(self,X):
        X=np.asarray(X)
            
        
        if len(X.shape)==1: # 1 data point
            Strength=self.run_Profiling(X)
        else:

            Strength=np.apply_along_axis( self.run_Profiling,1,X)

        #print RMSE    
        return Strength*self.ismax     


class AlloyCooking_Profiling2:
    '''
    AlloyCooking_Profiling2: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(2*3600,4*3600)),('Time2',(2*3600,6*3600)),('Temp1',(225,300)),('Temp2',(300,350))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.626471140;
        myxmatrix=0.00067;
        myiSurfen=0.0774275920;
        myfSurfen=0.184312062;
        myRadsurfenchange=3.90761800e-09;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        eng.addpath(r'F:\Dropbox\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange)

        temp=np.asarray(strength)
        return temp[0][1]
        
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Strength=self.run_Profiling(X)
        else:
            Strength=np.apply_along_axis( self.run_Profiling,1,X)

        #print RMSE    
        return Strength*self.ismax     

class AlloyCooking_Profiling3:
    '''
    AlloyCooking_Profiling2: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('Time1',(1*3600,4*3600)),('Time2',(2*3600,4*3600)),('Temp1',(250,300)),('Temp2',(300,350))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 150
        self.ismax=1
        self.name='AlloyCooking_Profiling'
        
    def get_data(self,mystr):
        data = load_svmlight_file(mystr)
        return data[0], data[1]
    
    def run_Profiling(self,X):
        print X
        x1=X[0]
        x2=X[1]
        x3=X[2]
        x4=X[3]
        
        if x3<0.000001:
            x3=0.000001
        if x2<0.000001:
            x2=0.000001

        myEm=0.626471140;
        myxmatrix=0.0004056486;
        myiSurfen=0.0774275920;
        myfSurfen=0.184312062;
        myRadsurfenchange=3.90761800e-09;
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        eng.addpath(r'F:\Dropbox\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        
        #mycooktemp=matlab.double(np.array([x3,x4]))
        myCookTemp=matlab.double([x3,x4])
        myCookTime=matlab.double([x1,x2])
        strength=eng.PrepNuclGrowthModel_MultipleStages(myxmatrix,myCookTemp,myCookTime,myEm,myiSurfen,myfSurfen,myRadsurfenchange)

        temp=np.asarray(strength)
        return temp[0][1]
        
    def func(self,X):
        X=np.asarray(X)
        
        if len(X.shape)==1: # 1 data point
            Strength=self.run_Profiling(X)
        else:
            Strength=np.apply_along_axis( self.run_Profiling,1,X)

        #print RMSE    
        return Strength*self.ismax  
        
class AlloyKWN_Fitting:
    '''
    AlloyKWN_Fitting: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 4
        
        if bounds == None: 
            self.bounds = OrderedDict([('myEM',(0.35,0.8)),('iSurfen',(0.01,0.1)),('fsurfen',(0.16,0.2)),('radsurfenchange',(8e-10,5e-9))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        self.ismax=-1
        self.name='AlloyKWN_Fitting'
        
    
    def run_Evaluate_KWN(self,X):
        print X
        
        #import numpy as np
        import matlab.engine
        import matlab
        eng = matlab.engine.start_matlab()
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting',nargout=0)
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting\data',nargout=0)
        eng.addpath(r'P:\02.Sharing\Vu_Sunil_Santu\Alloy_Paul\parameters_fitting\BO-matlab-code',nargout=0)

        temp=matlab.double(X.tolist())
        myEM=temp[0][0]
        myiSurfen=temp[0][1]
        myfSurfen=temp[0][2]
        myradchange=temp[0][3]
        """
        myEM=matlab.double([0.4]);
        myiSurfen=0.07;
        myfSurfen=0.17;
        myradchange=0.0000000041;
        """
        RMSE=eng.Evaluating_Alloy_Model_wrt_FourParameters(myEM,myiSurfen,myfSurfen,myradchange)


        return RMSE
        
    def func(self,X):
        X=np.asarray(X)
            
        
        if len(X.shape)==1: # 1 data point
            RMSE=self.run_Evaluate_KWN(X)
        else:

            RMSE=np.apply_along_axis( self.run_Evaluate_KWN,1,X)

        #print RMSE    
        return RMSE*self.ismax
        
class DeepLearning_MLP_MNIST:

    '''
    SVR_function: function 
    
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,  bounds=None,sd=None):
        self.input_dim = 7
        
        if bounds == None:  # n_node: 512, dropout 0.2, 512, 0.2, 10 # learning rate, decay, momentum
            self.bounds = OrderedDict([('n_node1',(100,1000)),('dropout1',(0.01,0.5)),('n_node2',(100,500)),('dropout2',(0.01,0.5)),
                                        ('lr',(0.01,1)),('decay',(1e-8,1e-5)),('momentum',(0.5,1))])
        else: 
            self.bounds = bounds
        
        self.min = [(0.)*self.input_dim]
        self.fmin = 1
        self.ismax=1
        self.name='DeepLearning_MLP_MNIST'
        
    
    def run_MLP_MNIST(self,X,X_train,Y_train,X_test,Y_test):
        #print X
        # Para: 512, dropout 0.2, 512, 0.2, 10
        
        from keras.models import Sequential
        from keras.layers.core import Dense, Dropout, Activation
        from keras.optimizers import SGD, Adam, RMSprop
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 10
        
        model = Sequential()
        x1=np.int(X[0])
        model.add(Dense(x1, input_shape=(784,)))
        
        model.add(Activation('relu'))
        
        temp=np.int(X[1]*100)
        x2=temp*1.0/100
        
        model.add(Dropout(x2))
        #model.add(Dense(512))
        
        x3=np.int(X[2])
        model.add(Dense(x3))
        model.add(Activation('relu'))
        
        temp=np.int(X[3]*100)
        x4=temp*1.0/100
        
        model.add(Dropout(x4))
        
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        #model.summary()
        
        # learning rate, decay, momentum
        sgd = SGD(lr=X[4], decay=X[5], momentum=X[6], nesterov=True)
        
        
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=0, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        return score[1]
    def func(self,X):
        
        np.random.seed(1337)  # for reproducibility
        
        from keras.datasets import mnist

        from keras.utils import np_utils
    
        X=np.asarray(X)
        
        batch_size = 128
        nb_classes = 10
        nb_epoch = 10

        # the data, shuffled and split between train and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        #print(X_train.shape[0], 'train samples')
        #print(X_test.shape[0], 'test samples')
        
        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        

        if len(X.shape)==1: # 1 data point
            Accuracy=self.run_MLP_MNIST(X,X_train,Y_train,X_test,Y_test)
        else:

            Accuracy=np.apply_along_axis( self.run_MLP_MNIST,1,X,X_train,Y_train,X_test,Y_test)

        #print RMSE    
        return Accuracy*self.ismax     
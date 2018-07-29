# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:15:44 2017

@author: priyanka
"""

import numpy as np

class Perceptron:
    def __init__(self,lr=0.05,epoch=1000):
        self.lr=lr
        self.epoch=epoch
        
    def activation_function(self,x):
        if x > 0.0:
            return 1
        else :
            return 0 
            
    def train(self,training_data,target_data):
        self.training_data=training_data
        self.target_data=target_data
        self.weights=np.random.rand(self.training_data.shape[1]+1,1) # initializing random weights
        print self.weights.shape
        self.training_data=np.column_stack((self.training_data,np.ones((self.training_data.shape[0])))) #initialising bias as last row
        for _ in range(self.epoch):
            for xi,yi in zip(self.training_data,self.target_data): 
                xi=np.atleast_2d(xi)
                result=np.dot(xi,self.weights)
                output=self.activation_function(result)
                print output
                loss=yi-output
                update=self.lr*loss
                self.weights+=(update*xi).T
        print "trainning over"
        
    def predict(self,test_data):
        new_test_data=np.column_stack((test_data,np.ones((test_data.shape[0])))) 
        result=np.dot(new_test_data,self.weights)
        output=self.activation_function(result)
#        output=[]
#        for ri in result:
#            o=self.activation_function(ri)
#            output.append(o)
        print "{} -> {}".format(test_data,output)



p=Perceptron()
data=np.array([[0,0],[0,1],[1,0],[1,1]])
output=np.array([[0],[0],[0],[1]])
p.train(data,output)
p.predict(np.array([[1,0]]))

        
        
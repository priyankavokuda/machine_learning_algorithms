# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 17:11:55 2017

@author: priyanka
"""

import numpy as np
import matplotlib.pyplot as plt


num_points=100

x=np.atleast_2d(np.linspace(0,num_points/10,num_points))
y=np.atleast_2d(map(lambda y:0.9*y+0.5,x))

x=x.T
y=y.T

offset=float(num_points)/float(75)
print offset
noise=np.atleast_2d(np.random.uniform(-offset,offset,num_points))


y_=y+noise.T
#x_=x+noise

plt.plot(x,y_,"bo")
plt.plot(x,y,"g-")

plt.show()

A=np.column_stack((x,np.ones((x.shape))))
w=np.dot(np.linalg.pinv(A),y)

print"real", w
print 
w_init=np.random.rand(2,1)
w_init=np.array(w_init)

epoch=0
lr=0.01
j=0
diff=100
for i in range(num_points):
    if (np.abs(diff)>0.0001):
        w_temp=np.zeros((w_init.shape))
        diff=(w_init[0,0]*x[i,0]+w_init[1,0])-y[i,0]
        j_m=diff*x[i,0]
        j_c=diff
        w_temp[0,0]=w_init[0,0]-lr*j_m
        w_temp[1,0]=w_init[1,0]-lr*j_c
        w_init=w_temp
        #print w_init
print "vector", w_init
print

w_init=np.random.rand(2,1)
w_init=np.array(w_init)

cost=100
for i in range(10000):
    if (np.abs(cost)>=0.0000001):
        loss=np.dot(A,w_init)-y
        cost=np.sum(loss**2)/(2*num_points)
        gradient=np.dot(x.T,loss)/num_points
        w_init=w_init-lr*gradient
    
    
print "matrix",w_init    

         
    

   
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 12:08:04 2017

@author: priyanka
"""

import numpy as np
import matplotlib.pyplot as plt

noise=np.random.uniform(5,10,100)
A_=np.linspace(0,10,100)
x_cos=np.cos(A_)*2*np.pi
b=x_cos+noise

A_=np.atleast_2d(A_).T
b=np.atleast_2d(b).T



plt.title("data")
plt.plot(A_,b,'bo')
plt.show()

print "least square fit"

A=np.column_stack((A_,np.ones((A_.shape))))
x1=np.dot(np.linalg.pinv(A),b)
plt.plot(A_,b,'bo')
plt.plot(A_,np.dot(A_,x1[0,0])+x1[1,0],'r-')

plt.show()

print "regularised least square"
poly_size=10
reg_constant=100
A=np.ones((A_.shape[0],poly_size+1))
for i in range(poly_size):
    A[:,i]=A_[:,0]**i
aTa=np.dot(A.T,A)

ab=np.dot(A.T,b)
xtx_inv=np.linalg.pinv(aTa-reg_constant*np.eye((aTa.shape[0])))
x2=np.dot(xtx_inv,ab)

plt.plot(A_,b,'bo')
plt.plot(A_,np.dot(A,x2),'g-')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:22:14 2021
@author: katharinaenin
"""
import numpy as np

# Set condition case
case = 1

folder = 'Example1/'

if case == 1:
    print("Use case 1.")
    
    case = 1 
    n = 10
    m = 600 
    number_of_edges = 7
    
    P_time0 = np.zeros((number_of_edges,n))
    Q_time0 = np.zeros((number_of_edges,n))
    for i in range(0,number_of_edges):
        for j in range(0,n):
            P_time0[i][j] = 60
            Q_time0[i][j] = 100

    np.savetxt(folder + 'P_time0.dat', P_time0)
    np.savetxt(folder + 'Q_time0.dat', Q_time0)


    eps = np.zeros((1,m))
    P_initialnode = np.zeros((1,m))
    Q_initialnode = np.zeros((1,m))
    
    for i in range(0,m):
        # eps steigt nach der Hälfte der Zeit schlagartig an
        if i < m/2:
            eps[0,i] = 0
        else:
            eps[0,i] = 5
        
        P_initialnode[0,i] = 60
        Q_initialnode[0,i] = 100
    
    np.savetxt(folder + 'eps.dat', eps)
    np.savetxt(folder + 'P_initialnode.dat', P_initialnode)
    np.savetxt(folder + 'Q_initialnode.dat', Q_initialnode)

elif case == 2:
    print("Use case 2.")
else: 
    print("No case provided.")
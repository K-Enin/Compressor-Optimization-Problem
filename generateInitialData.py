#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:22:14 2021
@author: katharinaenin
"""
import numpy as np

# Set condition case
case = 1
Q_choice = 1

folder = 'Example6/'

if case == 1:
    print("Use case 1.")
    
    case = 1 
    n = 10
    m = 600 
    number_of_edges = 7
    
    P_time0 = np.zeros((number_of_edges,n))
    Q_time0 = np.zeros((number_of_edges,n))
    
    # P for time 0 
    steps = (60 - 43)/(5 * n - 5)
    init = 60
    for i in [0,1,2,4,6]:
        for j in range(0,n):
            if j != 0:
                init = init - steps
            P_time0[i][j] = init
    P_time0[3,:] = P_time0[2,:]
    P_time0[5,:] = P_time0[4,:]
    
    
    # Q for time 0
    if Q_choice == 0:
        for i in range(0,number_of_edges):
            for j in range(0,n):
                if i in [2,3,4,5]:
                    Q_time0[i][j] = 200
                else: 
                    Q_time0[i][j] = 400
                    
    # Q is decreasing                
    elif Q_choice == 1:
        steps_Q = (550 - 400)/(5 * n - 5) #550 - 400 war bisher das beste
        init_Q = 550
        for i in [0,1,2,4,6]:
            for j in range(0,n):
                if j != 0:
                    init_Q = init_Q - steps_Q
                Q_time0[i][j] = init_Q
        Q_time0[2,:] = Q_time0[2,:]/2
        Q_time0[3,:] = Q_time0[2,:]
        Q_time0[4,:] = Q_time0[4,:]/2
        Q_time0[5,:] = Q_time0[4,:]
        

    np.savetxt(folder + 'P_time0.dat', P_time0)
    np.savetxt(folder + 'Q_time0.dat', Q_time0)

    eps = np.zeros((1,m))
    P_initialnode = np.zeros((1,m))
    Q_initialnode = np.zeros((1,m))
    
    # Set initial nodes
    for i in range(0,m):
        # eps steigt nach der Hälfte der Zeit an
        if i < m/2:
            eps[0,i] = 0
        else:
            eps[0,i] = 0
        
        P_initialnode[0,i] = 60
        Q_initialnode[0,i] = Q_time0[6,n-1]
    
    np.savetxt(folder + 'eps2.dat', eps)
    np.savetxt(folder + 'P_initialnode.dat', P_initialnode)
    np.savetxt(folder + 'Q_lastnode.dat', Q_initialnode)

elif case == 2:
    print("Use case 2.")
else: 
    print("No such case exists.")
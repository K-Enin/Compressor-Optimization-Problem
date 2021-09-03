#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:47:12 2021
Model for solving Power-Flow Equations
@author: katharinaenin
"""

from scipy.optimize import fsolve
#import math
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos
from numpy import sin

## We have the following nodes
## [N1, N2, N3, N4, N5, N6, N7, N8, N9]
nodes_list = np.arange(0,9)

B = np.loadtxt(open("bus_B.csv", "rb"), delimiter=";")
G = np.loadtxt(open("bus_G.csv", "rb"), delimiter=";")

if (np.allclose(B, B.T, rtol=1e-05, atol=1e-08) == False):
    raise Exception("Matrix B is not symmetric.")
    
if (np.allclose(G, G.T, rtol=1e-05, atol=1e-08) == False):
    print("Matrix G is not symmetric.")

initial = np.loadtxt(open("powergrid_initial.csv", "rb"), delimiter = ";", skiprows = 1)

m = 600


################################################################
### Generate real and Reactive Power at Node 5 for all t=0,...,m 

def generate_P_at_N5():
    """
    Generate Real Power at Node N5
    """
    P_at_N5 = np.zeros(m)
    
    P_old = -90
    P_new = -180
    margin_lower = m/2
    margin_upper = m/2+50
    slope = (abs(P_new)-abs(P_old))/(margin_upper - margin_lower)
    c = abs(P_old) - slope*margin_lower
    
    for t in range(0,m):
        if t < margin_lower:
            P_at_N5[t] = P_old
        elif margin_lower <= t < margin_upper:
            P_at_N5[t] = -(slope*t + c)
        else:
            P_at_N5[t] = P_new
    
    return P_at_N5

def generate_Q_at_N5():
    """
    Generate Reactive Power at Node N5
    """
    Q_at_N5 = np.zeros(m)
    
    Q_old = -30
    Q_new = -60
    margin_lower = m/2
    margin_upper = m/2+50
    slope = (abs(Q_new)-abs(Q_old))/(margin_upper - margin_lower)
    c = abs(Q_old) - slope*margin_lower
    
    for t in range(0,m):
        if t < margin_lower:
            Q_at_N5[t] = Q_old
        elif margin_lower <= t < margin_upper:
            Q_at_N5[t] = -(slope*t + c)
        else:
            Q_at_N5[t] = Q_new
    
    return Q_at_N5


P_at_N5 = generate_P_at_N5() 
Q_at_N5 = generate_Q_at_N5()


################################################################
### Solve Power Flow Equations

def power_model(z):
    """
    Power flow equation for one time step
    """
    # real power
    # P = z[0] # N1 unknown
    # # reactive power
    # Q = z[1:4] # N1, N2, N3
    # # voltage amplitude
    # V_ampl = z[4:10] # N4, N5, N6, N7, N8, N9
    # # phase
    # Phi = z[10:18] # N2, N3, N4, N5, N6, N7, N8, N9
    #for t in range(0,m):
    sum_V_k = 0
    sum_phi_k = 0
    sum_P_k = 0
    sum_Q_k = 0
    F = np.zeros(0)
    for node_k in nodes_list:
        
        if initial[node_k][2] == -999:
            P_k = z[0 + sum_P_k]
            sum_P_k = sum_P_k + 1
        else:
            P_k = initial[node_k][2]
        
        if initial[node_k][2] == -999:
            Q_k = z[1 + sum_Q_k]
            sum_Q_k = sum_Q_k + 1
        else:
            Q_k = initial[node_k][2] 

        if initial[node_k][2] == -999:
            V_k = z[4 + sum_V_k]
            sum_V_k = sum_V_k + 1
        else:
            V_k = initial[node_k][2]
        
        if initial[node_k][2] == -999:
            phi_k = z[10 + sum_phi_k]
            sum_phi_k = sum_phi_k + 1
        else:
            phi_k = initial[node_k][2] 
        
        sum_V_j = 0
        sum_phi_j = 0
        sum_comp_P = 0
        sum_comp_Q = 0
        for node_j in nodes_list:
            # V
            if initial[node_j][2] == -999: 
                V_j = z[4 + sum_V_j] 
                sum_V_j = sum_V_j + 1
            else:
                V_j = initial[node_j][2]
            
            # Phi
            if initial[node_j][3] == -999: 
                phi_j = z[10 + sum_phi_j] 
                sum_phi_j = sum_phi_j + 1
            else:
                phi_j = initial[node_j][3]
            
            # Computation
            sum_comp_P = sum_comp_P + abs(V_k)*abs(V_j)*(G[node_k, node_j]*cos(phi_k - phi_j) + B[node_k, node_j]*sin(phi_k - phi_j))
            sum_comp_Q = sum_comp_Q + abs(V_k)*abs(V_j)*(G[node_k, node_j]*sin(phi_k - phi_j) - B[node_k, node_j]*cos(phi_k - phi_j))
        
        F = np.append(F, [P_k - sum_comp_P, Q_k - sum_comp_Q])
    
    return F
        
        # Versuch 1
        # if node_k == 0: # slack
        #     print("Slack node: " + str(node_k))
        #     sum = 0
        #     for node_j in nodes_list:
        #         sum = sum + abs()*abs()*(G[node_k, node_j]*cos()+B[node_k, node_j]*cos())
        # elif node_k in [2,3,4]: # generator 
        #     print("Generator node: " + str(node_k))
        # elif node_k == 5: # special load node
        #     print("Special load node: " + str(node_k))
        # else:           # all other nodes
        #     print("Load node: " + str(node_k))
        
if __name__ == '__main__':
    plot_N5 = False
    if plot_N5 == True:
        P_at_N5 = generate_P_at_N5()
        plt.plot(np.arange(600),P_at_N5)
        Q_at_N5 = generate_Q_at_N5()
        plt.plot(np.arange(600),Q_at_N5)

    zGuess = np.zeros(2*9)
    z = fsolve(power_model,zGuess)
    
## CASADI Rootfinder

# z = cas.SX.sym('z',1)
# x = cas.SX.sym('x',1)
# g0 = cas.sin(x+z)
# g1 = cas.cos(x-z)
# g = cas.Function('g',[z,x],[g0,g1])
# G = cas.rootfinder('G','newton',g)
# G_sol = G(i0=0, i1=0)
# print(G)

## FSOLVE

# def my_function(z):
#     x=z[0]
#     y=z[1]
#     F=np.empty(2)
#     F[0]=x**2+y-5
#     F[1]=x**2+y**2-7
#     return F

# zGuess = np.array([1,1])
# z=optimize.fsolve(my_function,zGuess)
# print(z)

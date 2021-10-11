#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 20:47:12 2021
Model for solving Power-Flow Equations
@author: katharinaenin
"""

from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos
from numpy import sin

## We have the following nodes
## [N1, N2, N3, N4, N5, N6, N7, N8, N9]

nodes_list = np.arange(0,9)

B = np.loadtxt(open("bus_B.csv", "rb"), delimiter = ";")
G = np.loadtxt(open("bus_G.csv", "rb"), delimiter = ";")

if (np.allclose(B, B.T, rtol=1e-05, atol=1e-08) == False):
    raise Exception("Matrix B is not symmetric.")
    
if (np.allclose(G, G.T, rtol=1e-05, atol=1e-08) == False):
    raise Exception("Matrix G is not symmetric.")

initial = np.loadtxt(open("powergrid_initial.csv", "rb"), delimiter = ";", skiprows = 1)

m = 600
margin_lower = m/2
margin_upper = m/2+50


###################################################################
### Generate real and Reactive Power at Node 5 for all t = 0,...,m 
# Hier richtig!
def generate_P_at_N5():
    """
    Generate Real Power at Node N5
    """
    P_at_N5 = np.zeros(m)
    
    P_beg = -90
    P_end = -180
    slope = (abs(P_end)-abs(P_beg))/(margin_upper - margin_lower)
    c = abs(P_beg) - slope*margin_lower
    
    for t in range(0,m):
        if t < margin_lower:
            P_at_N5[t] = P_beg
        elif margin_lower <= t < margin_upper:
            P_at_N5[t] = -(slope*t + c)
        else:
            P_at_N5[t] = P_end
    
    return P_at_N5

def generate_Q_at_N5():
    """
    Generate Reactive Power at Node N5
    """
    Q_at_N5 = np.zeros(m)
    
    Q_beg = -30
    Q_end = -60
    slope = (abs(Q_end)-abs(Q_beg))/(margin_upper - margin_lower)
    c = abs(Q_beg) - slope*margin_lower
    
    for t in range(0,m):
        if t < margin_lower:
            Q_at_N5[t] = Q_beg
        elif margin_lower <= t < margin_upper:
            Q_at_N5[t] = -(slope*t + c)
        else:
            Q_at_N5[t] = Q_end
    
    return Q_at_N5


P_at_N5 = generate_P_at_N5() 
Q_at_N5 = generate_Q_at_N5()


################################################################
### Solve Power Flow Equations

def power_model(z):
    """
    Power flow equation for one time step
    """
    # P = z[0] # N1 unknown
    # Q = z[1:4] # N1, N2, N3
    # V_ampl = z[4:10] # N4, N5, N6, N7, N8, N9
    # Phi = z[10:18] # N2, N3, N4, N5, N6, N7, N8, N9
    sum_V_k = 0
    sum_phi_k = 0
    sum_P_k = 0
    sum_Q_k = 0
    F = np.zeros(0)
    time_t = 0
    for node_k in nodes_list:
        if initial[node_k][0] == -999:
            P_k = z[0 + sum_P_k]
            sum_P_k = sum_P_k + 1
        elif node_k == 4: # special load node
            P_k = P_at_N5[time_t]
        else:
            P_k = initial[node_k][0]
        
        if initial[node_k][1] == -999:
            Q_k = z[1 + sum_Q_k]
            sum_Q_k = sum_Q_k + 1
        elif node_k == 4: # special load node
            Q_k = Q_at_N5[time_t]
        else:
            Q_k = initial[node_k][1] 

        if initial[node_k][2] == -999:
            V_k = z[4 + sum_V_k]
            sum_V_k = sum_V_k + 1
        else:
            V_k = initial[node_k][2]
        
        if initial[node_k][3] == -999:
            phi_k = z[10 + sum_phi_k]
            sum_phi_k = sum_phi_k + 1
        else:
            phi_k = initial[node_k][3] 
        
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
            
            # Calculate Power Flow Equations
            sum_comp_P = sum_comp_P + abs(V_k)*abs(V_j)*(G[node_k, node_j]*cos(np.deg2rad(phi_k - phi_j)) + B[node_k, node_j]*sin(np.deg2rad(phi_k - phi_j)))
            sum_comp_Q = sum_comp_Q + abs(V_k)*abs(V_j)*(G[node_k, node_j]*sin(np.deg2rad(phi_k - phi_j)) - B[node_k, node_j]*cos(np.deg2rad(phi_k - phi_j)))
        
        F = np.append(F, [P_k - sum_comp_P, Q_k - sum_comp_Q])
    
    return F
        
if __name__ == '__main__':
    plot_N5 = False
    plot_N1 = False
    
    if plot_N5 == True:
        plt.figure(1)
        P_at_N5 = generate_P_at_N5()
        plt.plot(np.arange(600),P_at_N5)
        Q_at_N5 = generate_Q_at_N5()
        plt.plot(np.arange(600),Q_at_N5)

    Matrix_List = np.zeros((0,18))
    P_N1_beg, Q_N1_beg, P_N1_end, Q_N1_end = 70, -20, 170, -60 #P ist positiv!s
    sign = np.sign(P_N1_beg)
    test1 = []
    test2 = []
    
    zGuess = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Adapt guess to time t
    for t in range(0,m):
        if t % 50 == 0: print("This is t: " + str(t))
        time_t = t
    #     if t < margin_lower:
    #         zGuess = [P_N1_beg, Q_N1_beg, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #         test1.append(P_N1_beg)
    #         test2.append(Q_N1_beg)
    #     elif margin_lower <= t < margin_upper:
    #         slope_1 = (abs(P_N1_end)-abs(P_N1_beg))/(margin_upper - margin_lower)
    #         c_1 = P_N1_beg - sign*slope_1*margin_lower
    #         slope_2 = (abs(Q_N1_end)-abs(Q_N1_beg))/(margin_upper - margin_lower)
    #         c_2 = Q_N1_beg - sign*slope_2*margin_lower
    #         zGuess = [sign*slope_1*t + c_1, sign*slope_2*t + c_2 ,0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #         test1.append(sign*slope_1*t + c_1)
    #         test2.append(sign*slope_2*t + c_2)
    #     else:
    #         zGuess = [P_N1_end, Q_N1_end,0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #         test1.append(P_N1_end)
    #         test2.append(Q_N1_end)
            
        z = fsolve(power_model,zGuess)
        Matrix_List = np.vstack((Matrix_List, z))
    
    if plot_N1 == True:
        plt.figure(2)
        plt.plot(np.arange(600), Matrix_List[:,0]) # real power
        plt.plot(np.arange(600), Matrix_List[:,1]) # reactive power
        # a0, a1, a2 = 2, 5, 10
        a0, a1, a2 = 1, 2, 0.001
        Epsilon = a0*np.ones(m) + a1*Matrix_List[:,0] + a2*Matrix_List[:,0]*Matrix_List[:,0] # hier schaukelt es sich sehr stark hoch
        plt.figure(3)
        plt.plot(np.arange(600), Epsilon)
        plt.figure(4)
        plt.plot(np.arange(600), test1)
        plt.plot(np.arange(600), test2)





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

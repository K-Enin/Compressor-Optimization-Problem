#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:17:14 2021
Discretized NLP for compressor optimization with outer convexification.
@author: katharinaenin
"""

import numpy as np
import pandas as pd
import casadi as cas
import re
import itertools

length_of_pipe = 200; #universal length, constant in m
time_in_total = 10; #time in seconds


def get_ingoing_arcs(df, node):
    """
    Function for extracting ingoing arcs from Arcs.txt
    input: dataframe, node from which we want to get the ingoing arcs
    output: list of arcs
    """
    list_of_arcs = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][2] == str(node):
            list_of_arcs.append(int(df.iloc[i][2]))
            
    return list_of_arcs


def get_outgoing_arcs(df, node):
    """
    Function for extracting outgoing arcs from Arcs.txt
    input: dataframe, node from which we want to get the outgoing arcs 
    output: list of arcs
    """
    list_of_arcs = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][1] == str(node):
            list_of_arcs.append(int(df.iloc[i][0]))

    return list_of_arcs


def get_number_of_compressors_and_arcs(df):
    """
    Function for extracting number of compressors from Arcs.txt
    input: dataframe
    output: int
    """
    list_of_compressors = []
    number_of_arcs = df.shape[0] - 1 #substract 1 because of the slack
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if(bool(re.match("c[0-9]",df.iloc[i][j]))):
                list_of_compressors.append(df.iloc[i][j])
    
    unique_list_of_compressors = list(set(list_of_compressors))
    return number_of_arcs, unique_list_of_compressors    


def get_slack_connection_node(df):
    """
    Function for extracting node which is connected to slack in Arcs.txt
    input: dataframe
    output: int
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if(bool(re.match("s",df.iloc[i][j]))):
                return df.iloc[i][j-1]


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_initialnode, eps):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_initialnode, eps as np
    output: P, Q, alpha
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    m = np.shape(P_initialnode)[1] # Number of time steps
    dx = length_of_pipe/n;
    dt = time_in_total/m;
    df1 = pd.read_csv('Arcs.txt', header = None)
    number_of_arcs, list_of_compressors = get_number_of_compressors_and_arcs(df1)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors
    
    list_of_configs = [list(i) for i in itertools.product([0, 1], repeat=number_of_configs)]
    
    alpha = cas.MX.sym('alpha', m, number_of_configs)
    
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    
    w0 += [.5] * (m-1) * number_of_configs #list with 208 elements, which are all 0.5
    lbw += [0.] * (m-1) * number_of_configs #"
    ubw += [2.] * (m-1) * number_of_configs    
    
    ####################
    #### Condition 5 ###
    eps = np.loadtxt('eps.dat')
    slack_connection_node = get_slack_connection_node(df1) #slack_connection_node = 5
    slack_connection_node_out_arcs = get_outgoing_arcs(df,slack_connection_node) #6,7
    
    for j in slack_connection_node_out_arcs:
        if isinstance(df.iloc[j][2],int):
           true_slack_connection_node_out_arcs = df.iloc[j][2]
            
    P, Q = [],[]
    for i in range(0,number_of_arcs+1):
        P += [cas.MX.sym('P_{}_{}'.format(i, df.loc[i]), n, m)] # nx x nt matrix with entries like 'xi_p_0_(0,2)'
        Q += [cas.MX.sym('Q_{}_{}'.format(i, df.loc[i]), n, m)]
    
    list_ingoing_arcs = get_ingoing_arcs(df,slack_connection_node)
    for t in range(0,m):
        sum_of_Q = 0
        for j in  range(0,len(list_ingoing_arcs)):
            sum_of_Q = sum_of_Q + Q[j][n,t]

        g += sum_of_Q - Q[true_slack_connection_node_out_arcs][0,t] - eps[t]
        lbg += [0.]
        ubg += [0.]
    
    ####################
    #### Condition 6 ###
    
    
# We need function for sum up rounding

if __name__ == '__main__':
    # nur zum testen
    df = pd.read_csv('Arcs.txt', header = None)
    #slack_connection_node = get_slack_connection_node(df)
    aaa, list_of_compressors = get_number_of_compressors_and_arcs(df)
    list_of_ingoing_arcs = get_ingoing_arcs(df, 5)
    list_of_outgoing_arcs = get_outgoing_arcs(df, 5)
    
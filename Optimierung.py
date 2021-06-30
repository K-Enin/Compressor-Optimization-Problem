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

length_of_pipe = 200; # universal length, constant in m
time_in_total = 10; # time in seconds


def get_ingoing_edges(df, node):
    """
    Function for extracting ingoing edges from Edges.txt
    input: dataframe; node from which we want to get the ingoing edges
    output: list of edges (type: list of int)
    """
    list_of_edges = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][2] == str(node):
            list_of_edges.append(int(df.iloc[i][0]))
            
    return list_of_edges


def get_outgoing_edges(df, node):
    """
    Function for extracting outgoing arcs from Edges.txt
    input: dataframe; node from which we want to get the outgoing edges 
    output: list (int)
    """
    list_of_edges = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][1] == str(node):
            list_of_edges.append(int(df.iloc[i][0]))
    return list_of_edges


def get_all_nodes(df):
    """
    Function for extracting all nodes from Edges.txt
    input: dataframe
    output: list (int & str)
    """
    list_of_all_nodes = []
    for i in range(1,df.shape[0]):
        for j in range(1,3):
            try:
                integer_value = int(df.iloc[i][j])
                if integer_value not in list_of_all_nodes:
                    list_of_all_nodes.append(integer_value)
            except ValueError:
                if df.iloc[i][j] not in list_of_all_nodes:
                    list_of_all_nodes.append(df.iloc[i][j])
    return list_of_all_nodes


def get_end_edge_in_network(df):
    """
    Function for extracting end edge and end node in network from Edges.txt
    We assume that there is only one node, that has an outgoing edge which isn't
    further specified. We assume furthermore that is written at the bottom.
    input: dataframe 
    output: end edge (int), end node which is connected to that edge (int)
    """
    return df.shape[0]-1, df.iloc[df.shape[0]-1][2]


def get_starting_nodes_in_network(df):
    """
    Function for extracting starting nodes from Edges.txt
    input: dataframe 
    output: list of starting nodes (int)
    """
    starting_nodes = []
    all_nodes = get_all_nodes(df)
    list_df_col2 = df.iloc[:,2]
    list_df2 = []
    
    for entry in list_df_col2:
        try:
           list_df2.append(int(entry))
        except ValueError:
            list_df2.append(entry)
           
    for node in all_nodes:
        if node not in list_df2:
            starting_nodes.append(node)
    
    return starting_nodes


def get_starting_edges_in_network(df):
    """
    Function for extracting starting edges from Edges.txt
    input: dataframe 
    output: list of starting edges (int)
    """
    list_df1 = []
    starting_edges = []
    list_df_col1 = df.iloc[:,1]
    starting_nodes = get_starting_nodes_in_network(df)
    
    for entry in list_df_col1:
        try:
           list_df1.append(int(entry))
        except ValueError:
            list_df1.append(entry)
    
    for node in starting_nodes:
        for i in range(1, df.shape[0]):
            try:
                if int(list_df_col1[i]) == node:
                    starting_edges.append(int(df.iloc[i][0]))
            except ValueError:
                pass
    
    return starting_edges


def get_number_of_compressors_and_edges(df):
    """
    Function for extracting number of edges and all existing compressors from Edges.txt
    input: dataframe
    output: int, list (str)
    """
    list_of_compressors = []
    number_of_edges = df.shape[0] #substract 1? because of the slack
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if(bool(re.match("c[0-9]",df.iloc[i][j]))):
                list_of_compressors.append(df.iloc[i][j])
    unique_list_of_compressors = list(set(list_of_compressors))
    return number_of_edges, unique_list_of_compressors    


def get_slack_connection_node(df):
    """
    Function for extracting node which is connected to slack in Arcs.txt
    input: dataframe
    output: int
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if df.iloc[i][j] == "s":
                return int(df.iloc[i][j-1])


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
    df = pd.read_csv('Arcs.txt', header = None)
    number_of_edges, list_of_compressors = get_number_of_compressors_and_edges(df)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors
    
    list_of_configs = [list(i) for i in itertools.product([0, 1], repeat = number_of_configs)]
    
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
    
    P, Q = [],[]
    for i in range(0,number_of_edges):
        P += [cas.MX.sym('P_{}_{}'.format(i, df.loc[i]), n, m)] # nx x nt matrix with entries like 'xi_p_0_(0,2)'
        Q += [cas.MX.sym('Q_{}_{}'.format(i, df.loc[i]), n, m)]
    
    ####################
    #### Condition 2 ###
    
    # Idee: für jeden knoten mache eine Liste aus ingoing arcs und eine Liste aus outgoing arcs
    # wir brauchen die arc nummer
    nodes_list = get_all_nodes(df)
    starting_nodes = get_starting_nodes_in_network(df)
    get_outgoing_edges = []
    get_ingoing_edges = []
    
    # for t in range(0,m)
    # TODO: you have to filter out all unnecessary nodes from nodes_list
    for node in nodes_list:
        # if node not in starting_nodes: #and not slack?! and not ending node?!
        ingoing_edges = get_ingoing_edges(df,node)
        outgoing_edges = get_outgoing_edges(df,node)
        # Q_in 
        for t in range(0,m):
            sum_Q_in = 0
            sum_Q_out = 0
            for in_edge in ingoing_edges:
                sum_Q_out = sum_Q_out + Q[in_edge][n,t]
            for out_edge in outgoing_edges:
                sum_Q_in = sum_Q_in + Q[out_edge][0,t]
            
            g += sum_Q_in + sum_Q_out
            lbg += [0.]
            ubg += [0.]
        
    
    ####################
    #### Condition 5 ###
    eps = np.loadtxt('eps.dat')
    slack_connection_node = get_slack_connection_node(df) #5
    slack_connection_node_out_edges = get_outgoing_edges(df,slack_connection_node) #6,7
    list_ingoing_edges = get_ingoing_edges(df,slack_connection_node)
    
    for j in slack_connection_node_out_edges:
        if isinstance(df.iloc[j][2],int):
           true_slack_connection_node_out_edges = df.iloc[j][2]
    
    for t in range(0,m):
        sum_of_Q = 0
        for j in range(0,len(list_ingoing_edges)):
            sum_of_Q = sum_of_Q + Q[j][n,t]

        g += sum_of_Q - Q[true_slack_connection_node_out_edges][0,t] - eps[t]
        lbg += [0.]
        ubg += [0.]
    
    
    ####################
    #### Condition 6 ###
    last_edge, last_node_in_network = get_end_edge_in_network(df)
    for t in range(0,m):
        g += P[last_edge][0,t]
        lbg += -cas.inf
        ubg += 41
    
    
    
# We need function for sum up rounding

if __name__ == '__main__':
    # nur zum testen
    df = pd.read_csv('Edges.txt', header = None)
    slack_connection_node = get_slack_connection_node(df)
    starting_edges = get_starting_edges_in_network(df)
    starting_nodes = get_starting_nodes_in_network(df)
    # aaa, list_of_compressors = get_number_of_compressors_and_edges(df)
    # list_of_ingoing_edges = get_ingoing_edges(df, 5)
    # list_of_outgoing_edges = get_outgoing_edges(df, 5)
    # last_node = get_end_edge_in_network(df)
    # all_nodes = get_all_nodes(df)
    
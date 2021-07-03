#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:17:14 2021
Discretized NLP for compressor optimization with outer convexification.
@author: katharinaenin

Copyright 2021 Katharina Enin
"""

import casadi as cas
import configparser
import itertools
import numpy as np
import pandas as pd
import re

# Read Config Parameters in Configs.txt
config = configparser.ConfigParser()   
config.read('Configs.txt')

length_of_pipe = int(config['configs']['LengthPipe']) # universal length for all pipes (m)
time_in_total = int(config['configs']['TimeInTotal']) # time (sec)
Lambda = float(config['configs']['Lambda'])
D = int(config['configs']['Diameter'])
min_pressure_last_node = int(config['configs']['MinPressureAtLastNode']) 
a = int(config['configs']['FluxSpeed']) 
a_square = a*a
# u = int(config['configs']['PressureIncreaseAtCompressor']) #?

def df_to_int(df):
    """
    Function for extracting integers encapsulated in strings 
    into real integers, e.g. '3' becomes 3
    input: dataframe with all strings
    output: adjusted dataframe with int and strings
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            try:
                df.iloc[i][j] = int(df.iloc[i][j])
            except:
                pass


def get_ingoing_edges(df, node):
    """
    Function for extracting ingoing edges for specific node
    input: dataframe, node from which we want to get the ingoing edges
    output: list of edges (int)
    """
    list_of_edges = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][2] == node:
            list_of_edges.append(df.iloc[i][0])
                
    return list_of_edges


def get_outgoing_edges(df, node):
    """
    Function for extracting outgoing edges for specific node
    input: dataframe, node from which we want to get the outgoing edges
    output: list of edges (int)
    """
    list_of_edges = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][1] == node:
            list_of_edges.append(df.iloc[i][0])

    return list_of_edges


def get_all_nodes(df):
    """
    Function for extracting all nodes from Edges.txt
    input: dataframe
    output: list of nodes (int & str)
    """
    list_of_all_nodes = []
    for i in range(1,df.shape[0]):
        for j in range(1,3):
            if df.iloc[i][j] not in list_of_all_nodes:
                list_of_all_nodes.append(df.iloc[i][j])

    return list_of_all_nodes


def get_end_node_in_network(df):
    """
    Function for extracting end node in network from Edges.txt
    assumption: there is only one end node.
    input: dataframe 
    output: last node in network (int)
    """
    all_nodes = get_all_nodes(df)
    end_nodes = []
    list_df2 = df.iloc[:,1].tolist()

    for node in all_nodes:
       if node not in list_df2:
           end_nodes.append(node)
    
    end_nodes.remove('s') # remove slack node, which has no outgoing edge (only ingoing)
    
    return end_nodes


def get_end_edge_in_network(df):
    """
    Function for extracting last edge in network from Edges.txt
    assumption: there is only one end edge attached to single end node. 
    thus break for loop if it is found
    input: dataframe 
    output: last edge (int)
    """
    end_node = get_end_node_in_network(df)
    list_df2 = df.iloc[:,2].tolist()
    
    for i, node in enumerate(list_df2): 
        if node == end_node[0]:
            return df.iloc[i][0]


def get_starting_nodes_in_network(df):
    """
    Function for extracting starting nodes from Edges.txt
    input: dataframe 
    output: list of starting nodes (int)
    """
    starting_nodes = []
    all_nodes = get_all_nodes(df)
    list_df2 = df.iloc[:,2].tolist()

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
    starting_edges = []
    list_df1 = df.iloc[:,1].tolist()
    starting_nodes = get_starting_nodes_in_network(df)

    for node in starting_nodes:
        for i in range(1, df.shape[0]):
            if list_df1[i] == node:
                starting_edges.append(df.iloc[i][0])
                break
    
    return starting_edges


def get_list_of_compressors(df):
    """
    Function for extracting number of all existing compressors from Edges.txt
    input: dataframe
    output: list of compressors (str)
    """
    list_of_compressors = []
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if(bool(re.match("c[0-9]",str(df.iloc[i][j])))):
                list_of_compressors.append(df.iloc[i][j])
    unique_list_of_compressors = list(set(list_of_compressors))
    
    return unique_list_of_compressors    


def get_slack_connection_node(df):
    """
    Function for extracting the node which is connected to slack bus in Arcs.txt
    assumption: only one node is connected to slack bus
    input: dataframe
    output: int
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if df.iloc[i][j] == "s":
                return df.iloc[i][j-1]
            

def get_all_edges_without_slack_edge(df):
    """
    Function for returning list of edges without the edge connected to the slack bus
    (Useful for condition 2)
    input: dataframe
    output: list
    """
    list_df1 = df.iloc[:,2].tolist()
    list_of_edges = []
    for i, node in enumerate(list_df1):
        if node != 's' and i != 0:
            list_of_edges.append(df.iloc[i][0])
    
    return list_of_edges


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_initialnode, eps, Arcsfile):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_initialnode, eps
           Arcsfile as string file -> ('Arcs.txt')
    output: P, Q, alpha
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    m = np.shape(P_initialnode)[1] # Number of time steps
    dx = length_of_pipe/n
    dt = time_in_total/m
    df = pd.read_csv(Arcsfile, header = None)
    number_of_edges = df.shape[0]
    list_of_compressors = get_list_of_compressors(df)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors
    
    alpha = cas.MX.sym('alpha', m, number_of_configs)
    
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    
    w0 += [.5] * (m-1) * number_of_configs
    lbw += [0.] * (m-1) * number_of_configs
    ubw += [2.] * (m-1) * number_of_configs    
    
    P, Q = [],[]
    u = []
    for i in range(0, number_of_edges): #sicher 1
        P += [cas.MX.sym('P_{}_{}'.format(i, df.loc[i]), n, m)] # nx x nt matrix with entries like 'xi_p_0_(0,2)'
        Q += [cas.MX.sym('Q_{}_{}'.format(i, df.loc[i]), n, m)]
    
    for com in list_of_compressors:
        u += [cas.MX.sym('u_{}'.format(com), m)]
    
    ####################
    #### Condition 1 ###
    # PDE constraint
    
    all_edges = get_all_edges_without_slack_edge(df)  
    for edge in all_edges: 
        for t in range(0,m-1):
            for j in range(1,n-1):
                g += P[edge][j,t+1] - 0.5*(P[edge][j+1,t] + P[edge][j-1,t]) + \
                    dt/(2*dx)*(Q[edge][j+1,t] - Q[edge][j-1,t])
                lbg += [0.]
                ubg += [0.]
                g += Q[edge][j,t+1] - 0.5*(Q[edge][j+1,t] + Q[edge][j-1,t]) + \
                    dt/(2*dx)*(a_square)*(P[edge][j+1,t] - P[edge][j-1,t]) + \
                        dt*Lambda/(4*D)*(Q[edge][j+1,t]*abs(Q[edge][j+1,t])/P[edge][j+1,t] + \
                                         Q[edge][j-1,t]*abs(Q[edge][j-1,t])/P[edge][j-1,t]) 
                lbg += [0.]
                ubg += [0.]
                
    ####################
    #### Condition 2 ###
    # Node property
    
    nodes_list = get_all_nodes(df)
    slack_connection_node = get_slack_connection_node(df) # 5
    starting_nodes = get_starting_nodes_in_network(df)
    end_node = get_end_node_in_network(df)
    
    # sum q_in = sum q_out
    # Filter out all unnecessary nodes from nodes_list 
    # which are starting nodes, ending nodes, slack attached nodes
    for node in starting_nodes:
        nodes_list.remove(node)
    nodes_list.remove(slack_connection_node)
    nodes_list.remove(end_node)
    
    for node in nodes_list:
        ingoing_edges = []
        outgoing_edges = []
        ingoing_edges = get_ingoing_edges(df,node)
        outgoing_edges = get_outgoing_edges(df,node)
 
        for t in range(0,m):
            sum_Q_in = 0
            sum_Q_out = 0
            for in_edge in ingoing_edges:
                sum_Q_out = sum_Q_out + Q[in_edge][n,t]
            for out_edge in outgoing_edges:
                sum_Q_in = sum_Q_in + Q[out_edge][0,t]
            
            g += sum_Q_in - sum_Q_out
            lbg += [0.]
            ubg += [0.]

    # p_node = p_pipe
    # additionaly filter out compressors of nodes_list
    # The pressure values at the end of all arcs connected to the same node must be equal
    for node in list_of_compressors:
        nodes_list.remove(node)
        
    for node in nodes_list: 
        ingoing_edges = []
        outgoing_edges = []
        ingoing_edges = get_ingoing_edges(df, node)
        ingoing_edges = get_outgoing_edges(df, node)
        
        for t in range(0,m):
            for edge_in in ingoing_edges:
                for edge_out in outgoing_edges:
                    g += P[edge_in][n,t] - P[edge_out][0,t]
                    lbg += [0.]
                    ubg += [0.] 
            
    ####################
    #### Condition 3 ###
    # Properties at compressor station    
    
    # SOS1 constraint
    g += [cas.mtimes(alpha, cas.DM.ones(number_of_configs))]
    lbg += [1.] * (m) # ? unsure about m
    ubg += [1.] * (m)

    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_configs)]
    
    for com in list_of_compressors:
        ingoing_edge = get_ingoing_edges(df, com) # in our model there is only one ingoing edge to compressor
        outgoing_edge = get_outgoing_edges(df, com) # same holds for outgoing edge
        for t in range(0,m):
            g += a_square*(P[ingoing_edge][n,t] - P[outgoing_edge][0,t]) - sum(alpha[t,s]*c[s][com]*u[com][t] 
                                                                    for s in range(0,number_of_configs))
            lbg += [0.]
            ubg += [0.]
            g += u[com][t] - sum(alpha[t,s]*u[com][t]*c[s][com]
                                 for s in range(0,number_of_configs)) 
            lbg += [0.]
            ubg += [+cas.inf]
    
    ####################
    #### Condition 5 ###
    # Properties at slack connection node
    
    eps = np.loadtxt('eps.dat')
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
    # Properties at last node
    last_edge, last_node_in_network = get_end_edge_in_network(df)
    for t in range(0,m):
        g += P[last_edge][0,t]
        lbg += -cas.inf
        ubg += min_pressure_last_node
    
    J = 0
    # Objective function
    for t in range(0,m):
        J = J + sum(alpha[t,s]*c[s][com]*u[com][t]
                    for s in range(0, number_of_configs) for com in range(0,number_of_compressors))

    # Create NLP dictionary
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return nlp, lbw, ubw, lbg, ubg, w0
    return nlp
    
# def extract solution
    
# def sum up rounding


if __name__ == '__main__':
    # nur zum testen
    df = pd.read_csv('Edges.txt', header = None)
    df_to_int(df)
    list_of_edges = get_all_edges_without_slack_edge(df)
    # end_edge =  get_end_edge_in_network(df)
    # list_of_compressors = get_list_of_compressors(df)
    # all_nodes = get_all_nodes(df)
    
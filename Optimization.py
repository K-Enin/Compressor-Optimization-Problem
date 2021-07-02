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

# Read Config Parameter
config = configparser.ConfigParser()   
config.read('Configs.txt')

length_of_pipe = config['configs']['LengthPipe'] # universal length for all pipes (m)
time_in_total = config['configs']['TimeInTotal'] # time (sec)
Lambda = config['configs']['Lambda']
D = config['configs']['Diameter']
min_pressure_last_node = config['configs']['MinPressureAtLastNode'] 
flux_speed = config['configs']['FluxSpeed'] 

def df_to_int(df):
    """
    Function for turning integers encapsulated in strings 
    into real integers, e.g. '3' becomes 3
    input: dataframe
    output: adjusted dataframe
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            try:
                df.iloc[i][j] = int(df.iloc[i][j])
            except:
                pass


def get_ingoing_edges(df, node):
    """
    Function for extracting ingoing edges from Edges.txt
    input: dataframe, node from which we want to get the ingoing edges
    output: list of edges (type: list of int)
    """
    list_of_edges = []
    for i in range(0, df.shape[0]):
        if df.iloc[i][2] == node:
            list_of_edges.append(df.iloc[i][0])
                
    return list_of_edges


def get_outgoing_edges(df, node):
    """
    Function for extracting outgoing arcs from Edges.txt
    input: dataframe; node which we want to get the outgoing edges from
    output: list (int)
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
    output: list (int & str)
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
    Assumption: There is only one end node.
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
    Assumption: There is only one end edge attached to single end node. Break for loop
    if it is found
    input: dataframe 
    output: last edge (int)
    """
    end_node = get_end_node_in_network(df)
    list_df2 = df.iloc[:,2].tolist()
    
    for i, item in enumerate(list_df2): 
        if item == end_node[0]:
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
    output: int, list (str)
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
    Function for extracting node which is connected to slack in Arcs.txt
    assumption: only one node is conneted to slack
    input: dataframe
    output: int
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if df.iloc[i][j] == "s":
                return df.iloc[i][j-1]


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_initialnode, eps, Arcsfile):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_initialnode, eps (all as numpy object)
           Arcs as string file -> ('Arcs.txt')
    output: P, Q, alpha
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    m = np.shape(P_initialnode)[1] # Number of time steps
    dx = length_of_pipe/n
    dt = time_in_total/m
    df = pd.read_csv(Arcsfile, header = None)
    number_of_edges = df.shape[0]
    list_of_compressors = get_list_of_compressors(df)
    number_of_configs = 2**len(list_of_compressors)
    
    # braucht man das?
    list_of_configs = [list(i) for i in itertools.product([0, 1], repeat = number_of_configs)]
    
    alpha = cas.MX.sym('alpha', m, number_of_configs)
    
    w = []
    w0 = []
    lbw = []
    ubw = []
    g = []
    lbg = []
    ubg = []
    
    w0 += [.5] * (m-1) * number_of_configs # list with 208 elements, which are all 0.5
    lbw += [0.] * (m-1) * number_of_configs # "
    ubw += [2.] * (m-1) * number_of_configs    
    
    P, Q = [],[]
    for i in range(0,number_of_edges):
        P += [cas.MX.sym('P_{}_{}'.format(i, df.loc[i]), n, m)] # nx x nt matrix with entries like 'xi_p_0_(0,2)'
        Q += [cas.MX.sym('Q_{}_{}'.format(i, df.loc[i]), n, m)]
    
    ####################
    #### Condition 1 ###
    # PDE constraint

    

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
    last_edge, last_node_in_network = get_end_edge_in_network(df)
    for t in range(0,m):
        g += P[last_edge][0,t]
        lbg += -cas.inf
        ubg += min_pressure_last_node
    
    
    
# We need function for sum up rounding

if __name__ == '__main__':
    # nur zum testen
    df = pd.read_csv('Edges.txt', header = None)
    # df_to_int(df)
    # end_edge =  get_end_edge_in_network(df)
    # list_of_compressors = get_list_of_compressors(df)
    # all_nodes = get_all_nodes(df)
    
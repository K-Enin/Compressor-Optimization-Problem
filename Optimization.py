#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 16:17:14 2021
Discretized NLP for compressor optimization with partial outer convexification.
@author: katharinaenin

Copyright 2021 Katharina Enin
"""

import casadi as cas
import configparser
import itertools
import numpy as np
import pandas as pd
import re

# Set folder name, provide like this: 'Example1/'
folder = 'Example1/'

# Read Config Parameters in Configs.txt
config = configparser.ConfigParser()   
config.read(folder + 'Configs.txt')

length_of_pipe = int(config['configs']['LengthPipe']) # here: universal length for all pipes (in m)
time_in_total = int(config['configs']['TimeInTotal']) # time (in sec)
Lambda = float(config['configs']['Lambda'])
D = int(config['configs']['Diameter'])
min_pressure_last_node = int(config['configs']['MinPressureAtLastNode']) 
a = int(config['configs']['FluxSpeed']) 
a_square = a*a

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
    Function for extracting all nodes from Edges.txt (inclusive slack & compressor)
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
    assumption: there is only one end node
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
    assumption: there is only one end edge attached to single end node, 
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
    Function for extracting all existing compressors from Edges.txt
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
    Function for extracting the node which is connected to slack bus in Edges.txt
    assumption: only one node is connected to slack bus
    input: dataframe
    output: int
    """
    for i in range(0,df.shape[0]):
        for j in range(0,df.shape[1]):
            if df.iloc[i][j] == "s":
                return df.iloc[i][j-1]
            

def get_slack_connection_edge(df):
    """
    Function for extracting the edge which is connected to slack bus in Edges.txt
    assumption: only one edge is connected to slack bus
    input: dataframe
    output: int
    """
    slack_connection_node = get_slack_connection_node(df)   
    list_of_outgoing_edges = []
    list_of_outgoing_edges = get_outgoing_edges(df, slack_connection_node) #6,7
    for edge in list_of_outgoing_edges:
        if df.iloc[edge,2] == 's':
            return df.iloc[edge,0]
        

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


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_initialnode, eps, Edgesfile):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_initialnode, eps
           Edgesfile as string file -> ('Edges.txt')
    output: nlp, lbw, ubw, lbg, ubg, w0
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    m = np.shape(P_initialnode)[0] # Number of time steps
    dx = length_of_pipe/n # in (m)
    dt = time_in_total/m  # in (s) 
    df = pd.read_csv(Edgesfile, header = None)
    df_to_int(df)
    number_of_edges = df.shape[0] - 1  # minus header
    list_of_compressors = get_list_of_compressors(df)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors
    slack_edge = get_slack_connection_edge(df) # 6
    
    # variables
    w = []
    w0 = []
    lbw = []
    ubw = []

    # constraints
    g = []
    lbg = []
    ubg = []   
    
    P, Q = [],[]
    u = []
    
    ################################
    ### Set initial conditions #####

    alpha = cas.MX.sym('alpha', m, number_of_configs)
    w += [cas.reshape(alpha, -1, 1)] # reshape zeilwenweise -> becomes one row
    w0 += [.5] * m * number_of_configs
    lbw += [0.] * m * number_of_configs
    ubw += [2.] * m * number_of_configs 
    
    # SOS1 constraint
    g += [cas.mtimes(alpha, cas.DM.ones(number_of_configs))] #600 x 2 und 2 x 1 -> 600 x 1
    print(type(g))
    lbg += [1.] * (m)
    ubg += [1.] * (m)
    
    starting_edges = get_starting_edges_in_network(df)
    
    for com in range(0, number_of_compressors):
        u += [cas.MX.sym('u_{}'.format(com), m, 1)]
        w += [cas.reshape(u[com], -1, 1)]

    w0 += [0.] * m * number_of_compressors
    lbw += [0.] * m * number_of_compressors
    ubw += [+cas.inf] * m * number_of_compressors #([0.] * number_of_compressors + [+cas.inf] * (number_of_compressors * (m-1))) * 2
    
    for edge in range(0, number_of_edges): # do not exclude slack edge
        P += [cas.MX.sym('P_{}'.format(edge), n, m)] # nxm matrix with entries like 'Q_2' (2 is edge)
        Q += [cas.MX.sym('Q_{}'.format(edge), n, m)]
        w += [cas.reshape(P[edge], -1, 1), cas.reshape(Q[edge], -1, 1)]
        
        # Set initial conditions
        if edge in starting_edges:
            # gehe zuerst zeilenweise durch P
            w0 += [P_initialnode]
            lbw += [P_initialnode] 
            ubw += [P_initialnode]
            for t in range(0,m):
                for j in range(1,n):
                    if t == 0:
                        w0 += [P_time0[edge,j]]
                        lbw += [P_time0[edge,j]]
                        ubw += [P_time0[edge,j]]
                    else: 
                        w0 += [0.]
                        lbw += [-100000]#[-cas.inf] # vielleicht lieber 0?
                        ubw += [100000]#[+cas.inf]

            # gehe anschließend zeilenweise durch Q
            w0 += [Q_initialnode]
            lbw += [Q_initialnode] 
            ubw += [Q_initialnode]
            for t in range(0,m):
                for j in range(1,n):
                    if t == 0:
                        w0 += [Q_time0[edge,j]]
                        lbw += [Q_time0[edge,j]]
                        ubw += [Q_time0[edge,j]]
                    else: 
                        w0 += [0.]
                        lbw += [-100000] #[-cas.inf]
                        ubw += [100000] #[+cas.inf]
        else:
            # gehe zuerst zeilenweise durch P
            for t in range(0,m):
                for j in range(0,n):
                    if t == 0:
                        w0 += [P_time0[edge,j]]
                        lbw += [P_time0[edge,j]]
                        ubw += [P_time0[edge,j]]
                    else:
                        w0 += [0.]
                        lbw += [-100000]#[-cas.inf]
                        ubw += [100000]#[+cas.inf]
            # gehe anschließend zeilenweise durch Q
            for t in range(0,m):
                for j in range(0,n):
                    if t == 0:
                        w0 += [Q_time0[edge,j]]
                        lbw += [Q_time0[edge,j]]
                        ubw += [Q_time0[edge,j]]
                    else:
                        w0 += [0.]
                        lbw += [-100000]#[-cas.inf]
                        ubw += [100000]#[+cas.inf]  
        
    ####################
    #### Condition 1 ###
    # PDE constraint
    
    all_edges = get_all_edges_without_slack_edge(df)
    
    for edge in range(0, len(all_edges)):
        for t in range(0, m-1):
            # P_plus = P[edge][2:,t]
            # P_minus = P[edge][:-2,t]
            # Q_plus = Q[edge][2:,t]
            # Q_minus = Q[edge][:-2,t]
            g += [P[edge][1:-1,t+1] - 0.5*(P[edge][2:,t] + P[edge][:-2,t]) - \
                      dt/(2*dx)*(Q[edge][:-2,t] - Q[edge][2:,t])]

            lbg += [0.] * (n-2)
            ubg += [0.] * (n-2)
            
            g += [Q[edge][1:-1,t+1] - 0.5*(Q[edge][2:,t] + Q[edge][:-2,t]) - \
                     dt/(2*dx)*(a_square)*(P[edge][:-2,t] - P[edge][2:,t]) + \
                         dt*Lambda/(4*D)*(Q[edge][2:,t]*cas.fabs(Q[edge][2:,t])/P[edge][2:,t] + \
                                           Q[edge][:-2,t]*cas.fabs(Q[edge][:-2,t])/P[edge][:-2,t])]
            lbg += [0.] * (n-2)
            ubg += [0.] * (n-2)
                
    # ####################
    # #### Condition 2 ###
    # # Node property
    
    
    # nodes_list = get_all_nodes(df)
    # slack_connection_node = get_slack_connection_node(df) # 5
    # starting_nodes = get_starting_nodes_in_network(df)
    # end_node = get_end_node_in_network(df)
    
    # # sum q_in = sum q_out
    # # Filter out all unnecessary nodes from nodes_list 
    # # which are starting nodes, ending nodes, slack attached nodes
    # for node in starting_nodes:
    #     nodes_list.remove(node)
    # nodes_list.remove(slack_connection_node)
    # nodes_list.remove(end_node[0])
    # nodes_list.remove('s')
    
    # for node in nodes_list:
    #     ingoing_edges = []
    #     outgoing_edges = []
    #     ingoing_edges = get_ingoing_edges(df,node)
    #     outgoing_edges = get_outgoing_edges(df,node)
 
    #     sum_Q_in = np.zeros((1,m))
    #     sum_Q_out = np.zeros((1,m))
    #     for in_edge in ingoing_edges:
    #         sum_Q_out = sum_Q_out + Q[in_edge][n-1,:]
    #     for out_edge in outgoing_edges:
    #         sum_Q_in = sum_Q_in + Q[out_edge][0,:]
        
    #     g += [(sum_Q_in - sum_Q_out).reshape((-1,1))] # 2 x 1
    #     lbg += [0.,] * m
    #     ubg += [0.,] * m

    # # p_node = p_pipe
    # # additionaly filter out compressors of nodes_list
    # # The pressure values at the end of all arcs connected to the same node must be equal
    # for node in list_of_compressors:
    #     nodes_list.remove(node)
    # nodes_list.append(slack_connection_node)   
    # print(nodes_list)  
    # for node in nodes_list: 
    #     ingoing_edges = []
    #     outgoing_edges = []
    #     ingoing_edges = get_ingoing_edges(df, node)
    #     outgoing_edges = get_outgoing_edges(df, node)
        
    #     for edge_in in ingoing_edges:
    #              for edge_out in outgoing_edges:
    #                  g += [(P[edge_in][n-1,:] - P[edge_out][0,:]).reshape((-1,1))]
    #                  lbg += [0.] * m
    #                  ubg += [0.] * m
            
    # ####################
    # #### Condition 3 ###
    # # Properties at compressor station

    # # matrix of all configurations
    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_configs)]
    
    # for j, com in enumerate(list_of_compressors):
    #     ingoing_edge = get_ingoing_edges(df, com) # in our model there is only one ingoing edge to compressor
    #     outgoing_edge = get_outgoing_edges(df, com) # same holds for outgoing edge

    #     if len(ingoing_edge) != 0 and len(outgoing_edge) != 0: #proof for security
    #         ingoing_edge = ingoing_edge[0]
    #         outgoing_edge = outgoing_edge[0]
    #         # g += [a_square*(P[outgoing_edge][0,:] - P[ingoing_edge][n-1,:]) - sum(alpha[:][s]*c[s][j]*u[j,:] 
    #         #                                                         for s in range(0, number_of_configs))]
    #         # Alternative zu oben
    #         summe = np.zeros((m,1))
    #         for  s in range(0, number_of_configs):
    #             summe = summe + c[s][j]*alpha[:,s]*u[j][:]
    #         print("Condition 3: " + str(len(g)))
    #         g += [(a_square*(P[outgoing_edge][0,:] - P[ingoing_edge][n-1,:]) - summe.reshape((1,-1))).reshape((-1,1))]
    
            
    #         lbg += [0.] * m
    #         ubg += [0.] * m
        
    #         #g += [u[j,:] - sum(alpha[:,s]*u[j][:]*c[s][j]
    #         #                      for s in range(0, number_of_configs))]
    #         # Alternative zu oben
    #         summe = np.zeros((m,1))
    #         for s in range(0, number_of_configs):
    #             summe = summe + alpha[:,s]*u[j][:]*c[s][j]

    #         g += [(u[j][:] - summe).reshape((-1,1))]
    #         lbg += [0.] * m
    #         ubg += [0.] * m
        
        
    # ####################
    # #### Condition 5 ###
    # # Properties at slack connection node
    
    # slack_connection_node_out_edges = get_outgoing_edges(df,slack_connection_node) #6,7
    # list_ingoing_edges = get_ingoing_edges(df,slack_connection_node) #4,5
    
    # # get edge 7, the one that is not connected to slack nodes
    # for j in slack_connection_node_out_edges:
    #     if isinstance(df.iloc[j][2],int):
    #        filtered_slack_connection_node_out_edges = df.iloc[j][2]
    
    # # assumption: we assume that there is only one further outgoing edge besides the
    # # the slack connection edge
    # sum_of_Q = np.zeros((1,m))
    # for j in range(0,len(list_ingoing_edges)):
    #     sum_of_Q = sum_of_Q + Q[j][n-1,:]
    #     g += [(sum_of_Q - Q[filtered_slack_connection_node_out_edges][0,:] - eps[:].reshape((1,-1))).reshape((-1,1))]
    #     lbg += [0.] * m
    #     ubg += [0.] * m
    
    # ####################
    # #### Condition 6 ###
    # # Properties at last node
    # end_edge = get_end_edge_in_network(df) # edge 7

    # g += [a_square*P[end_edge][n-1,:].reshape((-1,1))]
    # lbg += [min_pressure_last_node] * m
    # ubg += [+cas.inf] * m
    
    ###########################
    #### Objective function ###
    J = 0
    # J = cas.MX.zeros(1,1)
    for t in range(0,m):
        J = J + sum(alpha[t,s]*c[s][com]*u[com][t]
                    for s in range(0, number_of_configs) for com in range(0, number_of_compressors)) # umschreiben!!!
    
    # Alternative zu oben
    # sum1 = np.zeros((m,1))
    # for s in range(0, number_of_configs):
    #     sum2 = np.zeros((m,1))
    #     for j in range(0, number_of_compressors):
    #         sum2 = sum2 + c[s][j]*u[:][j]
    #     sum1 = sum1 + alpha[:,s]*sum2
    
    # Create NLP dictionary
    parameters = [m, n, number_of_compressors, number_of_configs, number_of_edges]
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g) #dimension missmatch

    return parameters, nlp, lbw, ubw, lbg, ubg, w0
    
# def extract solution
def extract_solution(sol, parameters):
    """
    Function for setting up the NLP
    input: sol, parameters
    output: alpha, u, P, Q
    """
    m, n, number_of_compressors, number_of_configs, number_of_edges = parameters
    offset = 0
    alpha = np.array(cas.reshape(sol['x'][offset:offset + m * number_of_configs],
        m, number_of_configs))
    offset += m * number_of_configs
    u = np.array(cas.reshape(sol['x'][offset:offset + number_of_compressors * m],
        number_of_compressors, m))
    offset += number_of_compressors * m
    P, Q = [], []
    for i in range(0, number_of_edges):
        P += [np.array(cas.reshape(sol['x'][offset:offset + m * n], m, n))]
        offset += m * n
        Q += [np.array(cas.reshape(sol['x'][offset:offset + m * n], m, n))]
        offset += m * n
    
    return alpha, u, P, Q

# def sum up rounding


if __name__ == '__main__':
    P_initialnode = np.loadtxt(folder + 'P_initialnode.dat')
    Q_initialnode = np.loadtxt(folder + 'Q_initialnode.dat')
    P_time0 = np.loadtxt(folder + 'P_time0.dat')
    Q_time0 = np.loadtxt(folder + 'Q_time0.dat')
    eps = np.loadtxt(folder + 'eps.dat')
    Edgesfile = folder + 'Edges.txt'
    parameters, nlp, lbw, ubw, lbg, ubg, w0 = gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_initialnode, eps, Edgesfile)
    
    # Solving the problem with ipopt solver
    options = {'ipopt': {'tol': 1e-8}}
    solver = cas.nlpsol('solver', 'ipopt', nlp, options); 
    sol = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg) #evaluate function "solver" with initial conditions and boundries    
    alpha, u, P, Q = extract_solution(sol, parameters)

    
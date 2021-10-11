 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25
Discretized NLP for compressor optimization with Bonmin.
@author: katharinaenin

"""
import casadi as cas
import configparser
import itertools
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from time import process_time
import math
import scipy.io 

# Set folder name, provide like this: 'Example1/'
folder = 'Example12-2KomFinal/'

# Additional constrains couple over time: True - on, False - offs
additional_constraints = True
if additional_constraints:
    M1, M2 = 1/12, 1/12
    
# Constraint on P and on Q
lbP = 0 #lower boundary on P -> there is non negative pressure
ubP = 80 #80 #+cas.inf upper boundary on P -> der größte Wert bei Bonmin ist 76
lbQ = -200
ubQ = 600 #600 #+cas.inf  #max value in Q is 550

# Read Config Parameters in Configs.txt
config = configparser.ConfigParser()
config.read(folder + 'Configs.txt')

length_of_pipe = int(config['configs']['LengthPipe']) # here: universal length for all pipes (in m)
time_in_total = int(config['configs']['TimeInTotal']) # time (in sec)
Lambda = float(config['configs']['Lambda']) # here: Lambda is fix
D = int(config['configs']['Diameter']) # in m
min_pressure_last_node = int(config['configs']['MinPressureAtLastNode']) # in bar
a = int(config['configs']['FluxSpeed']) # in m/s
a_square = a*a
InitialGuess_P_beginEdge = int(config['InitialGuesses']['InitialGuess_P_beginEdge'])
InitialGuess_Q_beginEdge = int(config['InitialGuesses']['InitialGuess_Q_beginEdge'])
InitialGuess_P_middleEdge = int(config['InitialGuesses']['InitialGuess_P_middleEdge'])
InitialGuess_Q_middleEdge = int(config['InitialGuesses']['InitialGuess_Q_middleEdge'])
InitialGuess_P_endEdge = int(config['InitialGuesses']['InitialGuess_P_endEdge'])
InitialGuess_Q_endEdge = int(config['InitialGuesses']['InitialGuess_Q_endEdge'])


def df_to_int(df):
    """
    Function for extracting integers encapsulated in strings 
    into real integers, e.g. '3' becomes 3
    input: dataframe with all strings
    output: adjusted dataframe with int and strings
    """
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
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
    for i in range(df.shape[0]):
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
    for i in range(df.shape[0]):
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
    for i in range(df.shape[0]):
        for j in range(1,3):
            if df.iloc[i][j] not in list_of_all_nodes:
                list_of_all_nodes.append(df.iloc[i][j])

    return list_of_all_nodes


def get_end_node_in_network(df):
    """
    Function for extracting end node in network from Edges.txt
    assumption: there is only one end node, slack not included
    input: dataframe 
    output: last node in network (int)
    """
    all_nodes = get_all_nodes(df)
    end_nodes = []
    list_df2 = df.iloc[:,1].tolist()

    for node in all_nodes:
       if node not in list_df2:
           end_nodes.append(node)
    
    end_nodes.remove('s') # remove slack node
    
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
        for i in range(df.shape[0]):
            if list_df1[i] == node:
                starting_edges.append(df.iloc[i][0])
                break 
                # break, because starting node is only connected to one edge! 
    
    return starting_edges


def get_list_of_compressors(df):
    """
    Function for extracting all existing compressors from Edges.txt
    input: dataframe
    output: list of compressors (str)
    """
    list_of_compressors = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if(bool(re.match("c[0-9]",str(df.iloc[i][j])))):
                list_of_compressors.append(df.iloc[i][j])
    unique_list_of_compressors = list(set(list_of_compressors)) # make list unique
    
    return unique_list_of_compressors


def get_slack_connection_node(df):
    """
    Function for extracting the node which is connected to slack bus in Edges.txt
    assumption: only one node is connected to slack bus
    input: dataframe
    output: int
    """
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iloc[i][j] == "s": # works only if 's' is written at out_node
                return df.iloc[i][j-1] # return in_node
            

def get_slack_connection_edge(df):
    """
    Function for extracting the edge which is connected to slack bus in Edges.txt
    assumption: only one edge is connected to slack bus
    input: dataframe
    output: int
    """
    slack_connection_node = get_slack_connection_node(df)
    list_of_outgoing_edges = get_outgoing_edges(df, slack_connection_node) # 6,7
    for edge in list_of_outgoing_edges:
        if df.iloc[edge,2] == 's':
            return df.iloc[edge,0] # there is only one
        

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


def gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_lastnode, eps, Edgesfile):
    """
    Function for setting up the NLP
    input: P_time0, Q_time0, P_initialnode, Q_lastnode, eps
           Edgesfile as string file -> ('Edges.txt')
    output: nlp, lbw, ubw, lbg, ubg, w0
    """
    n = np.shape(P_time0)[1]       # Number of space steps
    print("This is n: " + str(n))
    m = np.shape(P_initialnode)[0] # Number of time steps
    print("This is m: " + str(m))
    dx = length_of_pipe/n # in (m)
    dt = time_in_total/m  # in (s) 
    
    # Sanity Checks
    # Is CFL fulfilled?
    if (dt*a_square > dx):
        raise Exception("CFL is not fulfilled.")
    # Dimensions are uniformy?
    if (np.shape(P_time0) != np.shape(Q_time0) or np.shape(P_initialnode)[0] != np.shape(Q_lastnode)[0]):
        raise Exception("Mitsmatch of dimension of Q_time0, P_time0 or P_initialnode, Q_lastnode.")
    
    df = pd.read_csv(Edgesfile, header = None)
    df = df.iloc[1:, :] # drop first row, which is the header
    df = df.reset_index(drop=True)
    df_to_int(df)
    number_of_edges = df.shape[0] #8
    number_of_edges_without_slack_edge = number_of_edges - 1
    list_of_compressors = get_list_of_compressors(df)
    number_of_compressors = len(list_of_compressors)
    number_of_configs = 2**number_of_compressors # = 2
    slack_edge = get_slack_connection_edge(df) # 7
    starting_edges = get_starting_edges_in_network(df)
    end_edge = get_end_edge_in_network(df)
    
    # variables
    w, w0, lbw, ubw  = [], [], [], []

    # constraints
    g, lbg, ubg = [], [], []  
    
    # Parameters we are looking for
    P, Q, u = [], [], []
    # alpha does not need vector initialization
    
    # Vector provided information, whether value should be discrete or not
    discrete = []
    
    ################################
    ### Set initial conditions #####
    
    # alpha
    alpha = cas.MX.sym('alpha', m, number_of_configs)
    w += [cas.reshape(alpha, -1, 1)] # reshape spaltenweise 
    w0 += [.5] * m * number_of_configs
    lbw += [0.] * m * number_of_configs
    ubw += [1.] * m * number_of_configs 
    discrete += [True] * m * number_of_configs 
    
    # u
    u = cas.MX.sym('u', m, number_of_compressors)
    w += [cas.reshape(u, -1, 1)]
    w0 += [0.] * m * number_of_compressors  
    lbw += [0.] * m * number_of_compressors
    ubw += [+cas.inf] * m * number_of_compressors
    discrete += [False] * m * number_of_compressors 
    
    # P, Q 
    for edge in range(number_of_edges_without_slack_edge):
        P += [cas.MX.sym('P_{}'.format(edge), n, m)]
        Q += [cas.MX.sym('Q_{}'.format(edge), n, m)]
        w += [cas.reshape(P[edge], -1, 1), cas.reshape(Q[edge], -1, 1)]
        
        if edge in starting_edges: 
            # go column-wise through P
            w0 += [*P_time0[edge]]
            lbw += [*P_time0[edge]] 
            ubw += [*P_time0[edge]]
            for t in range(1,m):
                w0 += [P_initialnode[t]] + [InitialGuess_P_beginEdge]*(n-1)
                lbw += [P_initialnode[t]] + [lbP]*(n-1)
                ubw += [P_initialnode[t]] + [ubP]*(n-1)
                
            # go column-wise through Q
            w0 += [*Q_time0[edge]] + [InitialGuess_Q_beginEdge]*n*(m-1)
            lbw += [*Q_time0[edge]] + [lbQ]*n*(m-1)
            ubw += [*Q_time0[edge]] + [ubQ]*n*(m-1)
            
        elif edge == end_edge: 
            # go column-wise through P
            w0 += [*P_time0[edge]] + [InitialGuess_P_endEdge]*n*(m-1)
            lbw += [*P_time0[edge]] + [lbP]*n*(m-1)
            ubw += [*P_time0[edge]] + [ubP]*n*(m-1)
            
            # go column-wise through Q
            w0 += [*Q_time0[edge]]
            lbw += [*Q_time0[edge]] 
            ubw += [*Q_time0[edge]]
            for t in range(1,m):
                w0 += [InitialGuess_Q_endEdge]*(n-1) + [Q_lastnode[t]]
                lbw += [lbQ]*(n-1) + [Q_lastnode[t]]
                ubw += [ubQ]*(n-1) + [Q_lastnode[t]]
        else: 
            # go column-wise through P
            w0 += [*P_time0[edge]] + [InitialGuess_P_middleEdge]*n*(m-1)
            lbw += [*P_time0[edge]] + [lbP]*n*(m-1)
            ubw += [*P_time0[edge]] + [ubP]*n*(m-1)
            
            # go column-wise through Q
            w0 += [*Q_time0[edge]] + [InitialGuess_Q_middleEdge]*n*(m-1)
            lbw += [*Q_time0[edge]] + [lbQ]*n*(m-1)
            ubw += [*Q_time0[edge]] + [ubQ]*n*(m-1)
    
    discrete += [False] * 2 * n * m * number_of_edges_without_slack_edge
    print("Initial conditions are set.")
    
    ####################
    #### Condition 1 ###
    # PDE constraint with Weymouth Equation
    bar_conv = 100000 # 1 bar = 100.000 (kg/(m*s^2))
    for edge in range(number_of_edges_without_slack_edge):
        for t in range(m-1):
            g += [P[edge][1:-1,t+1]*bar_conv/a_square - 0.5*(P[edge][2:,t]*bar_conv/a_square \
                  + P[edge][:-2,t]*bar_conv/a_square) \
                  - dt/(2*dx)*(Q[edge][:-2,t] - Q[edge][2:,t])]

            lbg += [0.] * (n-2)
            ubg += [0.] * (n-2)
            
            g += [Q[edge][1:-1,t+1] - 0.5*(Q[edge][2:,t] + Q[edge][:-2,t]) \
                  - dt/(2*dx)*(P[edge][:-2,t]*bar_conv - P[edge][2:,t]*bar_conv) \
                  + a_square*dt*Lambda/(4*D)*(Q[edge][2:,t]*cas.fabs(Q[edge][2:,t])/(P[edge][2:,t]*bar_conv) \
                  + Q[edge][:-2,t]*cas.fabs(Q[edge][:-2,t])/(P[edge][:-2,t]*bar_conv))]
            lbg += [0.] * (n-2)
            ubg += [0.] * (n-2)
    
    print("Condition 1 is set.")
    
    ####################
    #### Condition 2 ###
    # Node property
    
    ### sum q_in = sum q_out
    nodes_list = get_all_nodes(df)
    slack_connection_node = get_slack_connection_node(df) # 5
    starting_nodes = get_starting_nodes_in_network(df)
    end_node = get_end_node_in_network(df)
    
    # Filter out all unnecessary nodes from nodes_list 
    # which are starting nodes, ending nodes, slack attached nodes
    for node in starting_nodes:
        nodes_list.remove(node)
    nodes_list.remove(end_node[0]) # 6
    # slack connection node will be given an extra flow condition
    nodes_list.remove(slack_connection_node) # 5 
    nodes_list.remove('s')
    
    for node in nodes_list:
        ingoing_edges = get_ingoing_edges(df,node)
        outgoing_edges = get_outgoing_edges(df,node)
 
        sum_Q_out = sum(Q[in_edge][n-1,:] for in_edge in ingoing_edges)
        sum_Q_in = sum(Q[out_edge][0,:] for out_edge in outgoing_edges)
        
        g += [(sum_Q_in - sum_Q_out).reshape((-1,1))]
        lbg += [0.,] * m
        ubg += [0.,] * m
    
    #### p_node = p_pipe
    # remove compressor node
    for node in list_of_compressors:
        nodes_list.remove(node)
    nodes_list.append(slack_connection_node)
    
    for node in nodes_list: 
        ingoing_edges = get_ingoing_edges(df, node)
        outgoing_edges = get_outgoing_edges(df, node)

        for edge_in in ingoing_edges:
            for edge_out in outgoing_edges:
            # Only for edges which are last edge (thus slack edge)
                if edge_out != number_of_edges_without_slack_edge:
                    g += [(P[edge_in][n-1,:] - P[edge_out][0,:]).reshape((-1,1))]
                    lbg += [0.] * m
                    ubg += [0.] * m
    print("Condition 2 is set.")
    
    ####################
    #### Condition 3 ###
    # Properties at compressor station
    
    # SOS1 constraint
    g += [cas.mtimes(alpha, cas.DM.ones(number_of_configs))] #600 x 2 und 2 x 1 -> 600 x 1
    lbg += [1.] * (m)
    ubg += [1.] * (m)
    
    # list containing all compressor configurations
    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
    
    for j, com in enumerate(list_of_compressors):
        ingoing_edge = get_ingoing_edges(df, com) 
        outgoing_edge = get_outgoing_edges(df, com)

        # In our model there is one ingoing and one outgoing edge for every compressors
        # Angst vor zwei unterschiedlichen zusammenlaufenden Drücken?
        if len(ingoing_edge) == 1 and len(outgoing_edge) == 1: 
            ingoing_edge = ingoing_edge[0]
            outgoing_edge = outgoing_edge[0]

            sum_com = sum(c[s][j]*alpha[:,s]*u[:,j] for s in range(number_of_configs))

            g += [(P[outgoing_edge][0,:] - P[ingoing_edge][n-1,:] - sum_com.reshape((1,-1))).reshape((-1,1))]

            lbg += [0.] * m
            ubg += [0.] * m
    print("Condition 3 is set.")
    
    ####################
    #### Condition 4 ###
    # Properties at slack connection node
    
    list_outgoing_edges = get_outgoing_edges(df,slack_connection_node) # 6,7
    list_ingoing_edges = get_ingoing_edges(df,slack_connection_node) # 4,5
    
    # get edge 6, the one that is not connected to slack nodes
    # (!) Assumption: we assume that there is only one further outgoing edge besides
    # the slack connection edge
    for j in list_outgoing_edges:
        if j != number_of_edges_without_slack_edge: # not the last edge    
            filtered_slack_connection_node_out_edges = j # 6 

    print("What is slack connection node: " + str(filtered_slack_connection_node_out_edges))
    sum_of_Q = sum(Q[edge][n-1,:] for edge in list_ingoing_edges)
    
    g += [(sum_of_Q - Q[filtered_slack_connection_node_out_edges][0,:] - eps[:].reshape((1,-1))).reshape((-1,1))]
    lbg += [0.] * m
    ubg += [0.] * m
    print("Condition 4 is set.")
    
    ####################
    #### Condition 5 ###
    # Properties at last node

    g += [P[end_edge][n-1,:].reshape((-1,1))]
    lbg += [min_pressure_last_node] * m
    ubg += [+cas.inf] * m
    print("Condition 5 is set.")
    
    #############################
    #### Optional Condition 6 ###
    # Additional constraints that couple over time
    if additional_constraints == True:
        # Switch has to keep its state for at least time M1 from off to on
        for j in range(number_of_compressors):
            for k in range(m - math.floor(M1/dt)): # (m-6)
                rside = sum(c[con][j]*math.floor(M1/dt)*(-alpha[k,con] + alpha[k+1,con])
                    for con in range(number_of_configs))
                lside = sum(c[con][j]*alpha[e,con]
                    for con in range(number_of_configs) 
                    for e in range(k+1, k + math.floor(M1/dt))) 
                g += [lside - rside]
                lbg += [0]
                ubg += [+cas.inf]
        
        # Switch has to keep its state for at least time M2 from on to off
        for j in range(0, number_of_compressors):
            for k in range(0, m - math.floor(M2/dt)): # (m-6)
                rside = sum(c[con][j]*math.floor(M2/dt)*(alpha[k,con] - alpha[k+1,con])
                                                   for con in range(number_of_configs))
                lside = sum(1 - sum(c[con][j]*alpha[e,con] 
                        for con in range(number_of_configs)) 
                        for e in range(k+1, k + math.floor(M2/dt)))
                g += [lside - rside]
                lbg += [0]
                ubg += [+cas.inf]
        
        print("Additional conditions are set.")
    
    ###########################
    #### Objective function ###
    
    J = 0.5 * sum(u[t,j]**2 for t in range(m) for j in range(number_of_compressors))
    
    print("Objective function set.")
    
    # Create NLP dictionary
    parameters = [m, n, number_of_compressors, number_of_configs, number_of_edges_without_slack_edge, starting_edges, end_edge]
    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return parameters, nlp, lbw, ubw, lbg, ubg, w0, discrete


def extract_solution(sol, parameters):
    """
    Function for setting up the NLP
    input: sol, parameters
    output: alpha, u, P, Q
    """
    m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    offset = 0
    alpha = np.array(cas.reshape(sol['x'][offset:offset + m * number_of_configs],
        m, number_of_configs))
    offset += m * number_of_configs
    u = np.array(cas.reshape(sol['x'][offset:offset + number_of_compressors * m],
        m, number_of_compressors))
    offset += number_of_compressors * m
    P, Q = [], []
    for i in range(number_of_edges):
        P += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
        Q += [np.array(cas.reshape(sol['x'][offset:offset + m * n], n, m))]
        offset += m * n
    
    return alpha, u, P, Q

def plot_solution(alpha, u, P, Q, parameters):
    """
    Function for plotting the solution 
    input: alpha, u, P, Q
    """
    m, n, number_of_compressors, number_of_configs, number_of_edges, starting_edges, end_edge = parameters
    t = np.arange(0,m)

    c = [list(i) for i in itertools.product([0, 1], repeat = number_of_compressors)]
    
    # plot u
    plt.figure(1).clear()
    fig, axes  = plt.subplots(number_of_compressors, 1, num=2)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
        axes[i].step(t, u[:,i])
        axes[i].set_xlabel(r'time step $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))
    plt.savefig(folder + 'BONMINControl_u.png')
    
    # plot compressor on/off
    plt.figure(2).clear()
    # Vector provides which configuration is on for compressors
    configurations_on = []
    for time_step in range(m):
        case = False
        j = 0
        while case == False:
            if alpha[time_step, j] == 1:
                configurations_on.append(c[j])
                case = True
            j = j + 1
    configurations_on_vec = np.array(configurations_on)       
    fig, axes  = plt.subplots(number_of_compressors, 1, num=2)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_compressors):
         axes[i].step(t, configurations_on_vec[:,i])
         axes[i].set_xlabel(r'time step $t$')
         axes[i].set_ylabel(r'$Compressor{}$'.format(i))
    plt.savefig(folder + 'BONMINConf.png')
    
    # plot configurations
    plt.figure(3).clear()
    fig, axes  = plt.subplots(number_of_configs, 1, num=1)
    axes = axes.reshape((-1,)) if number_of_compressors > 1 else [axes]
    for i in range(number_of_configs):
        axes[i].step(t, alpha[:,i])
        axes[i].set_xlabel(r'time step $t$')
        axes[i].set_ylabel(r'Configs $alpha_{}$'.format(i))
    plt.savefig(folder + 'BONMINalpha.png')
    
    # plot inflow at first node
    plt.figure(4).clear()
    fig, axes = plt.subplots(num = 3)
    axes.plot(t, Q[starting_edges[0]][0,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'mass flow (kg/s), first node')
    plt.savefig(folder + 'BONMINflow_firstNode.png')
    
    # plot pressure at last node
    plt.figure(5).clear()
    fig, axes = plt.subplots(num = 4)
    axes.plot(t, P[end_edge][n-1,:])
    axes.set(xlabel = r'time step $t$', ylabel = 'pressure (bar), last node')
    plt.savefig(folder + 'BONMINpressure_lastNode.png')
    
    
if __name__ == '__main__':
     P_initialnode = np.loadtxt(folder + 'P_initialnode.dat')
     Q_lastnode = np.loadtxt(folder + 'Q_lastnode.dat')
     P_time0 = np.loadtxt(folder + 'P_time0.dat')
     Q_time0 = np.loadtxt(folder + 'Q_time0.dat')
     # eps = np.loadtxt(folder + 'eps2.dat')
     Edgesfile = folder + 'Edges.txt'
     mat_file = scipy.io.loadmat(folder + 'eps_file3.mat')
     eps = mat_file["eps"]
     eps[0,0] = round(eps[0,0],4) # to make things a little simpler 
     # eps = eps[:, 0:np.shape(P_initialnode)[0]]

     parameters, nlp, lbw, ubw, lbg, ubg, w0, discrete = gasnetwork_nlp(P_time0, Q_time0, P_initialnode, Q_lastnode, eps, Edgesfile)
    
     # Solving the problem with BONMIN solver    
     options = {'bonmin': {'max_wall_time': 1800}, 'discrete': discrete} # in seconds, 1800 s = 30 min
     t0_start = process_time()
     solver = cas.nlpsol('nlp_solver', 'bonmin', nlp, options);
     sol = solver(x0 = w0, lbx = lbw, ubx = ubw, lbg = lbg, ubg = ubg)  
     alpha, u, P, Q = extract_solution(sol, parameters)
     t0_stop = process_time()
     elapsed_time = t0_stop - t0_start
     print("Time elapsed during execution of POC :" + str(elapsed_time))
     plot_solution(alpha, u, P, Q, parameters)
     
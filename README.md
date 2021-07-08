# Compressor Optimization with Partial Outer Convexification (POC)

## General Information

In this code we consider a Gas-to-Power network with multiple compressors, which increase the pressure and a slack node, which converts gas into energy.
The aim of this code is to solve an optimization problem with the help of the POC method, such that all constraints (gas-coupling conditions, Euler Equation, slack & compressor condition) are satisfied and the objective function which is the sum of all pressure differences at the compressor nodes is minimized. \
Instead of Euler Equation we use Weymouth Equation which provides similar results and is better to control in the terms of the CFL condition. \
We discretize the Weymouth Equation with Lax Friedrich method. 

## Test setup

### Initial data
As initial conditions we provide the following scripts: Edges, P_time0, Q_time0, P_initialnode, Q_initialnode, eps. \
Edges.txt denotes all arcs in a directed graph, they are given as a list of directed entries.
P_time0, Q_time0 provide initial values for P, Q at each spatial step in each pipe at t=0, where the number of rows corresponds to the number of pipes and the number of columns to the number of time steps.
P_initialnode, Q_initialnode provides initial values for P, Q at starting node, where the number of columns corresponds to the time steps. 
Eps represents the flow that is taken from the slack bus for converting gas to power. Its number of rows are again equal the number of time steps.

### Run the Code
The initial data is provided in different folders. Here, for better illustration, you can also find pictures of the network.
In order to test different application just adapt the folder name in Optimization.py, line 19 and then run the file.

### Configs
In the Configs.txt you can find different config parameter, which can be change according to the desired network. 
So far the code can only work with universal parameters - hence all pipes have the same length, the same diameter, and the same frictional value.

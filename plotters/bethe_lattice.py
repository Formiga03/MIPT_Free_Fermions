import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def hamiltonian_creator_bethe(height: int, branching_factor: int = 3, j: float = 1, periodic: bool = True) -> np.ndarray:
    T = nx.balanced_tree(branching_factor, height)
    if periodic:
        node_tot = T.number_of_nodes()
        int_leaf = node_tot - 3**height
        leave_ind = [xx for xx in range(int_leaf, node_tot)]

        G2 = nx.random_regular_graph(2, 3**height)
        mapping = dict(zip(G2.nodes(), leave_ind))

        G3 = nx.relabel_nodes(G2, mapping)

        G_composed = nx.compose(T, G3)
    A_sparse = nx.to_scipy_sparse_array(G_composed)
    return A_sparse.todense()*j

def neel_state_creator_bethe(height: int) -> np.ndarray:
    lst = []
    for ii in range(height): 
        if ii%2==0: lst += [0]*(3**ii)
        if ii%2!=0: lst += [1]*(3**ii)
    return np.diag(lst)
    

def find_height(num:int):
    ii = 0
    while num!=0:
        num -= 3**ii
        ii+=1
    return ii-1

#print(find_height(hamiltonian_creator_bethe(15).shape[0]))


def num_of_nodes(order:int)->int:
    if order==0: return 1
    res = 1
    for ii in range(order):
        res += 3*(2**ii)
    return res

def hamiltonian_creator_bethe_per(height: int, j: float = 1) -> np.ndarray:
    G1 = nx.random_regular_graph(3, num_of_nodes(height))
    A_sparse = nx.to_scipy_sparse_array(G1)
    return A_sparse.todense()*j


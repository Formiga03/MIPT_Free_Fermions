""""
"""

import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm, sinm, cosm
from scipy import linalg
import scipy
import sys
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re

arg1 =  str(sys.argv[0])


np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def brick_structure_pair_evol(N:int, pair:bool)->np.ndarray:
    """

    N:
    pair:
    """
    A = np.zeros((N,N), dtype=complex)
    
    if pair:
        for jj in range(0, N, 2):
            A[jj:jj+2, jj:jj+2] = unitary_group.rvs(2)
    else:      
        for kk in range(1, N-1, 2):
            A[kk:kk+2, kk:kk+2] = unitary_group.rvs(2)
        A[np.ix_([0,N-1], [0,N-1])] = unitary_group.rvs(2)

    return A

def hamiltonian_creator_2D(L:int, j:float=1, periodic:bool=True)->np.ndarray:
    """
    Creates a tight-biding hamltonian for a 2D square lattice with dimensions LxL.
    - L: Size of a side o the square lattice.
    - j: Jump constant value.
    - periodic: Indicates of one wants to admit periodic boundaries (True) or not 
                (False).
    """
    dim = L*L
    mat_aux1 = np.zeros((L,L))
    h = np.zeros((dim, dim))

    # Hamiltonian for a tight-biding model of a string os length L
    mat_aux1 = np.diag(np.full(L-1, -j), 1)
    mat_aux1 += np.diag(np.full(L-1, -j), -1)

    if periodic:
        # horizontal boundary points
        mat_aux1[0, L-1] = -j  
        mat_aux1[L-1, 0] = -j

        # vertical boundary points
        h += np.diag(np.full(dim-2*L, -j), 2*L)
        h += np.diag(np.full(dim-2*L, -j), -2*L)

    # string component building in the same matrix
    for ii in range(L):
        h[ii*L:(ii+1)*L, ii*L:(ii+1)*L] = mat_aux1
    
    # site links building
    h += np.diag(np.full(dim-L, -j), L)
    h += np.diag(np.full(dim-L, -j), -L)

    return h

def neel_state_creator_2D(L:int)->np.ndarray:
    """
    Creates a Neel state matrix in a 2D square lattice such that every line has an 
    occupancy shifted one site in relation with the previous one.
    - L: Size of a side o the square lattice.
    """
    
    dim = L*L # number of sites in the lattice
    mat_aux1 = np.zeros((L,L)) # 1D Neel state with odd sittes occupied
    mat_aux2 = np.zeros((L,L)) # 1D Neel state with pair sittes occupied
    
    ns = np.zeros((dim, dim)) # ininatilized 2D Neel state matrix

    # 1D Neel states construction
    mat_aux1 = np.diag([x%2 for x in range(L)]) 
    mat_aux2 = np.diag([0 if i % 2 == 0 else 1 for i in range(L)])

    # 2D Neel states construction 
    for ii in range(L):
        if ii%2 == 0: ns[ii*L:(ii+1)*L, ii*L:(ii+1)*L] = mat_aux1
        if ii%2 == 1: ns[ii*L:(ii+1)*L, ii*L:(ii+1)*L] = mat_aux2

    return ns

def entanglement_entropy_calc(C:np.ndarray)->float:
    """
    Calculates the entaglement entropy of the correlation entropy between the two halves
    of the system.
    - C: Correlatrion matrix
    """
    L = np.shape(C)[0]

    BB = C
    AAA1 = BB[:L//2,:]
    AAA1 = AAA1[:,:L//2]
    Lambda_1= np.linalg.eigvals(AAA1)
    Lambda_2 = Lambda_1[Lambda_1>10**(-20)]
    S = - np.dot(Lambda_2, np.log(Lambda_2))
    Lambda_3 = 1-Lambda_1
    Lambda_4 = Lambda_3[Lambda_3>10**(-20)]
    S += - np.dot(Lambda_4, np.log(Lambda_4))
    
    return S

exp = False
Time = 4000
tt = list(range(0, 10000, 5))
Steps1 = [0, 1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + [Time-1]
if not exp: Steps1 = list(range(Time))

pmax = 0.4
pstep = 0.05
ps = np.arange(0, pmax+pstep, pstep)

for L in [4,6,8]:
    print("____________________________________________________________________________" 
          "_________________________")
    print("L="+str(L))

    dim = L*L

    t_v = hamiltonian_creator_2D(L)
    e1_t,v1_t = np.linalg.eigh(t_v)
    
    Data = []
    
    for Prob in ps: 
        print("-p=" + str(Prob))
        data =[]
        data1 = []
            
        for ll in tqdm(range(10)):

            C = neel_state_creator_2D(L)
            for jj in range(Time):
                K = brick_structure_pair_evol(dim, jj%2==0)

                C = np.dot(K,np.dot(C, np.conjugate(K).T)) 

                for k in range(dim):
                    xx = np.random.choice([0,1],p = [Prob,1-Prob])
                    d_k = np.zeros(dim)
                    d_k[k] += 1
                    if xx==0:
                        Prob1 = np.min([1,np.abs(np.real(C[k,k]))])
                        yy = np.random.choice([0,1],p = [Prob1,1-Prob1])
                        if yy ==0:
                            A_ik = C[:,k]
                            B_kj = C[k,:]
                            C_ikkj = np.einsum("i,j->ij", A_ik, B_kj)/C[k,k]
                            C = C +np.einsum("i,j->ij", d_k,d_k)  -C_ikkj

                        else:
                            A_ik = (d_k-C[:,k])
                            B_kj = (d_k-C[k,:])
                            C_ikkj = np.einsum("i,j->ij", A_ik, B_kj)/(1-C[k,k])
                            C = C -np.einsum("i,j->ij", d_k,d_k)  + C_ikkj
                
                # Entanglement Entropy colculation
                if jj in Steps1:
                    S = entanglement_entropy_calc(C)
                    data.append(np.real(S))
                
            data1.append(data.copy())    
            data.clear()

        Data.append(list(np.mean(data1, axis=0)))
        
    name = "data/EEqq_2D_"+str(L)+"x"+str(L)+"_p=("+str(pmax)+","+str(pstep)+")_T="+str(Time)
    if exp: name += "-exp"
    
    np.savetxt(name, np.real(Data))

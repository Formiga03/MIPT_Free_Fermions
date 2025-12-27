import sys
from random import random
import numpy as np
from scipy.stats import unitary_group
from scipy import linalg
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


def s_average(data:list[list], size:list[int], lmt:float)->list[list]:
    """
    flm_names: List of the names of the elements we want to calculate the entanglement 
               entropy time average of.
    lmt: Index from where we do the time average of the plateu.
    size: List of the sizes of the system in the files.
    """
    # initializing the necessary lists
    s_aveg = [] # output list
    lst_aux1 = [] # takes the whole data from a certain L
    lst_aux2 = [] # takes the list of S averaged in time

    for ii in range(len(data)):
        # taking data of one of the files
        lst_aux1 = data[ii]
        lst_aux2.append(size[ii])

        # calculation of the value of the time average of the entanglement entropy of 
        # of the platue for different values of measurement probability
        for jj in range(len(lst_aux1)-1):
            ss = np.average(lst_aux1[jj][lmt:])
            lst_aux2.append(ss)

        # inserting the results into the output list with the respective size as the 
        # starting element
        s_aveg.append(lst_aux2.copy())

        # cleaning auxiliary lists
        lst_aux1.clear()
        lst_aux2.clear()
    
    # sorting results by size and removing the size term from the lists
    s_aveg.sort()
    s_aveg = [x[1:] for x in s_aveg]

    return s_aveg




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

def brick_structure_odd_evol(N:int, iter_N:int)->np.ndarray:
    """
    Creates 
    N:
    iter_N:
    """
    A = np.identity(N, dtype=complex)
    
    if iter_N==1:
        for ii in range(0, N-1, 2):
            A[ii:ii+2,ii:ii+2] += unitary_group.rvs(2)

    elif iter_N%2==0:
        pos = iter_N - 2
        for ii in range(0, N-1, 2):
            pivot = ii + 1 + pos
            while pivot >= N: pivot = pivot - N
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

    elif iter_N%2==1 and iter_N!=1:
        N_blocks_odd = (N-1)//2 - 1
        pos = iter_N - 2

        A[np.ix_([0,N-1], [0,N-1])] = unitary_group.rvs(2)

        for jj in range(N_blocks_odd):
            pivot = pos + 1 + jj*2
            if pivot >= N-1: pivot = pivot - N + 2
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

    return A

def time_phase_disturbed_evol(ham: np.ndarray, time: float, amp: float = 1) -> np.ndarray:
    """
    ham: Hamiltonian matrix
    time: evolution time
    amp: amplitude factor (currently unused)
    """
    dim = len(ham)
    ham_exp = linalg.expm(-1j * ham * time)

    # Generate random diagonal values
    pot1_diag = np.array([np.random.uniform(-1, 1) for _ in range(dim)])

    # Create diagonal matrices with phase and imaginary components
    pot_mat = np.diag(np.exp(1j * pot1_diag))

    K = np.dot(pot_mat, ham_exp)

    return K

def monitored_circuit(C:np.ndarray, Prob:float)->None:
    """

    - C: Correlation matrix
    - Prob: Measurement probability
    """
    L = np.shape(C)[0]
    
    for k in range(L):
        xx = np.random.choice([0,1],p = [Prob,1-Prob])
        d_k = np.zeros(L)
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

# Time parameters
Time = 2000
t_step = 1
tt = list(range(0,Time,t_step))
Steps1 = [0, 1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + \
         [Time-1]
Steps1 = tt

# Probability parameters
pmax = 1
pstep = 0.1
ps = np.arange(0, pmax+pstep, pstep)

for L in [8, 16, 26]:
    print("__________________________________________________________________________" \
    "___________________________")
    print("L="+str(L))

    # Tight biding hamiltonian for a 1D lattice without potential and interactions
    t_v = np.diag(np.ones(L-1),1)
    t_v[L-1,0] = 1
    t_v += t_v.T

    # Energy eingenvalues and eigenvectores calculation
    e1_t,v1_t = np.linalg.eigh(t_v)

    if L%2==1: iters = list(range(1, L+1))

    Data = []

    for Prob in ps: 
        print("-p=" + str(Prob))

        data = []
        data1 = []

        for kk in tqdm(range(30)):
            C = np.zeros([L,L], dtype=complex)
            C = np.diag([x%2 for x in range(L)])
            
            count = 0

            for jj in range(Time):
                
                K = time_phase_disturbed_evol(t_v, jj)
                #print(len(K), len(K[0]))

                # Time evolution of the correlation matrix
                C = np.dot(K,np.dot(C, np.conjugate(K).T)) 
                #print(len(C),len(C[0]))

                # Monitoring process trought the lattice
                for k in range(L):
                    xx = np.random.choice([0,1],p = [Prob,1-Prob])
                    d_k = np.zeros(L)
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
        
        np.savetxt("data/EEqq_"+str(L)+"_p="+ str(Prob) +"_T=("+str(Time)+","+
                       str(t_step)+")_test", np.real(data1))
    


"""
Old:

if L%2==0: K = brick_structure_pair_evol(L, jj%2==0)
if L%2==1:
    if jj%L==0 and jj!=0: count += 1
    K = brick_structure_odd_evol(L, jj + 1 - L*count)

pars = [
    [5, 1, 2, 4],
    [7, 1, 2, 3],
    [9, 4, 5, 6],
    [2, 1, 2, 3],
    [2, 1, 4, 3],
    [2, 82, 2, 3],
    [2, 82, 2, 3]
]
ver = [list(dict.fromkeys([tuple(r[2:]) for r in pars])).index(tuple(row[2:])) for row
       in pars] # makes sure the time paramenters are all equal

"""

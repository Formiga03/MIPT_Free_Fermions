import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from scipy import linalg
from random import random
from multiprocessing import Pool
from numba import njit
import re
import os

###################################### Reader ##########################################

def read_data(filename:str)->list[list]:
    """
    Takes the values in a file and outputs a list of lists, in which each list has the
    elements of a single line.
    filename: path + name of the file which is to be read
    """
    dt = []
    with open(filename) as f:
        for line in f:
            dt.append(list(map(float, line.split())))
    return dt

def raw_data_reader(size:int, dim:str)->tuple[list]:
    """
    Gathers all files with the raw data from systems with size "size" and with different 
    values of measurement probability (mp) and put them all in a list. Gives as output a
    tuple in which the 1st elments is the data divided by different values of mp and the
    2nd the parameters of each element in data. 
    - size: int which represents the size of the system for which we want to extract the
            raw that from
    """
    # File names in data directory
    fl_names = os.listdir("data/")

    data = [] # raw data organized such that: 
              # [[iter1,iter2,...]:Meas.Prob.1, [[iter1,iter2,...]]::Meas.Prob.2,...]

    #       file name selection:
    fls = ["data/" + x for x in fl_names if  f'_{str(size)}_' in x and dim in x] 

    
    #       parameter extraction and verification:
    pars = [
        [
            float(f"{float(x):.2f}") if "." in x else int(x)
            for x in re.findall(r"-?\d+\.?\d*", s)
        ]
        for s in fls
    ]

    pars = [x[1:] for x in pars] # takes the system size parameter

    #       file reading
    for ii in range(len(fls)):
        lst = read_data(fls[ii])
        lst.insert(0,pars[ii][0])
        data.append(lst)

    #       sort by probability measurement value
    data.sort()
    data = [x[1:] for x in data] # takes the measurement probability parameter
    pars.sort()

    return data, pars

def s_average(data:list[str], lmt:float)->list[list]:
    """
    lmt: Index from where we do the time average of the plateu.
    """
    # initializing the necessary lists
    s_aveg = [] # output list

    for ii in range(len(data)):
        s_aveg.append(np.average(data[ii][lmt:]))     

    return s_aveg

################################# System Simulation ####################################

def entanglement_entropy_free(h:np.ndarray, subsystem:list, occ_modes:list)->tuple:

    L = h.shape[0]
    e, v = np.linalg.eigh(h)  # e: eigenvalues, v: eigenvectors (columns)

    # Build correlation matrix for the occupied modes
    v_occ = v[:, occ_modes]
    print(v_occ)
    C = v_occ @ v_occ.conj().T
    print("Correlation matrix C:\n", C)

    # Restrict to subsystem A
    CA = C[np.ix_(subsystem, subsystem)]
    z, _ = np.linalg.eigh(CA)

    # Clip eigenvalues to avoid numerical issues
    eps = 1e-12
    z = np.clip(z, eps, 1-eps)

    # Von Neumann entanglement entropy
    S = -np.sum(z * np.log(z) + (1-z) * np.log(1-z))
    return S, z

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
    pair:
    """
    A = np.identity(N, dtype=complex)
    
    if iter_N==1:
        for ii in range(0, N-1, 2):
            A[ii:ii+2,ii:ii+2] += unitary_group.rvs(2)
            print(ii)

    elif iter_N%2==0:
        pos = iter_N - 2
        for ii in range(0, N-1, 2):
            pivot = ii + 1 + pos
            print(pivot)
            if pivot >= N: pivot = 0
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

    if iter_N%2==1 and iter_N!=1:
        N_blocks_odd = (N-1)//2 - 1
        pos = iter_N -2

        A[np.ix_([0,N-1], [0,N-1])] = unitary_group.rvs(2)

        for jj in range(N_blocks_odd):
            pivot = pos + 1 + jj*2
            if pivot >= N-1: pivot = N - pivot
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

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

@njit
def monitor_process_numba(C, L, Prob):
    d_k = np.zeros(L)

    for k in range(L):
        # Random binary choice: 0 with Prob, 1 with (1 - Prob)
        xx = 0 if np.random.random() < Prob else 1

        d_k[:] = 0.0
        d_k[k] = 1.0

        if xx == 0:
            Prob1 = min(1.0, abs(np.real(C[k, k])))
            yy = 0 if np.random.random() < Prob1 else 1

            if yy == 0:
                A_ik = C[:, k]
                B_kj = C[k, :]
                outer = np.outer(A_ik, B_kj)
                C_ikkj = outer / C[k, k]

                C += np.outer(d_k, d_k)
                C -= C_ikkj
            else:
                A_ik = d_k - C[:, k]
                B_kj = d_k - C[k, :]
                outer = np.outer(A_ik, B_kj)
                C_ikkj = outer / (1.0 - C[k, k])

                C -= np.outer(d_k, d_k)
                C += C_ikkj

    return C


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
    #print(C)
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

def time_phase_disturbed_evol(ham: np.ndarray, amp: float = 1) -> np.ndarray:
    """
    ham: Hamiltonian matrix
    time: evolution time
    amp: amplitude factor (currently unused)
    """
    dim = len(ham)
    #ham_exp = linalg.expm(-1j * ham * time)

    # Generate random diagonal values
    pot1_diag = np.array([np.random.uniform(-1, 1) for _ in range(dim)])

    # Create diagonal matrices with phase and imaginary components
    pot_mat = np.diag(np.exp(1j * pot1_diag))

    K = np.dot(pot_mat, ham)

    return K
def run_one_simulation(args):
    L, Prob, Time, Steps1, t_v, C = args

    # Unique RNG seed
    seed = os.getpid() ^ id(args)
    np.random.seed(seed)  # works with numba-compatible RNG

    data = []

    for jj in range(Time):
        K = time_phase_disturbed_evol(t_v)
        C = np.dot(K, np.dot(C, np.conjugate(K).T))

        # --- Optimized monitoring process ---
        C = monitor_process_numba(C, L, Prob)

        # --- Entanglement Entropy calculation ---
        if jj in Steps1:
            S = entanglement_entropy_calc(C)
            data.append(np.real(S))

    return data


"""
def entanglement_entropy_calc(C:np.ndarray)->float:
    Calculates the entaglement entropy of the correlation entropy between the two halves
    of the system.
    - C: Correlatrion matrix
    L = np.shape(C)[0]
    #print(C)
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

def time_phase_disturbed_evol(ham: np.ndarray, amp: float = 1) -> np.ndarray:
 
    ham: Hamiltonian matrix
    time: evolution time
    amp: amplitude factor (currently unused)
  
    dim = len(ham)
    #ham_exp = linalg.expm(-1j * ham * time)

    # Generate random diagonal values
    pot1_diag = np.array([np.random.uniform(-1, 1) for _ in range(dim)])

    # Create diagonal matrices with phase and imaginary components
    pot_mat = np.diag(np.exp(1j * pot1_diag))

    K = np.dot(pot_mat, ham)

    return K


@njit
def update_C_fast_numba(C, Prob):
    Fast, Numba-safe C-update loop (no unsupported np.random.choice).
    L = C.shape[0]
    # Draw random numbers: 0 with Prob, 1 with 1-Prob
    xx = np.random.random(L) > Prob
    C_diag = np.real(np.diag(C))

    for k in range(L):
        if not xx[k]:  # Equivalent to xx[k] == 0
            Prob1 = min(1.0, abs(C_diag[k]))
            yy = 1 if np.random.random() > Prob1 else 0

            if yy == 0:
                A_ik = C[:, k]
                B_kj = C[k, :]
                denom = C[k, k]
                C_ikkj = np.outer(A_ik, B_kj) / denom
                C -= C_ikkj
                C[k, k] += 1.0
            else:
                A_ik = -C[:, k].copy()
                A_ik[k] += 1.0
                B_kj = -C[k, :].copy()
                B_kj[k] += 1.0
                denom = 1.0 - C[k, k]
                C_ikkj = np.outer(A_ik, B_kj) / denom
                C += C_ikkj
                C[k, k] -= 1.0

    return C

def run_one_simulation(args):

    L, Prob, Time, Steps1, t_v, C = args
    # Use process ID + memory reference to generate a unique seed
    seed = (os.getpid() ^ id(args)) % (2**32 - 1)
    np.random.seed(seed)

    data = []

    for jj in range(Time):
        # System evolution step
        K = time_phase_disturbed_evol(t_v)
        C = np.dot(K, np.dot(C, np.conjugate(K).T))

        # Fast Numba-accelerated update
        C = update_C_fast_numba(C, Prob)

        # Entanglement entropy calculation
        if jj in Steps1:
            S = entanglement_entropy_calc(C)
            data.append(np.real(S))
            
    return data

"""
import numpy as np, time
import matplotlib.pyplot as plt
from random import random
from multiprocessing import Pool
import os
import mkl 

mkl.set_num_threads( 16 )


################################# System Simulation ####################################
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

def time_phase_disturbed_evol(ham: np.ndarray, amp: float = 1.0) -> np.ndarray:
    dim = ham.shape[0]
    pot1_diag = np.random.uniform(-amp, amp, size=dim)
    pot_mat = np.exp(1j * pot1_diag)  # vector of complex phases
    return ham * pot_mat

def update_C(C, Prob):
    """
    Fast, Numba-safe update of the correlation matrix C.
    Avoids unsupported np.random.choice() by using np.random.random().
    """
    L = C.shape[0]
    xx = np.random.random(L) > Prob  # 0 with Prob, 1 with 1-Prob
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
    np.__config__.show()

    os.environ["MKL_NUM_THREADS"] = "6"
    os.environ["OMP_NUM_THREADS"] = "6"
    print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
    print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
    Prob, Time, Steps1, t_v, C = args

    # Unique RNG seed
    seed = os.getpid() ^ id(args)
    seed32 = int(seed % (2**32 - 1))
    np.random.seed(seed32)

    data = []

    for jj in range(Time):
 
        t0 = time.perf_counter()

        K = time_phase_disturbed_evol(t_v)
        C = np.dot(K, np.dot(C, np.conjugate(K).T))
        t1 = time.perf_counter()
        print(f"Time Evol. Op. Creation: {(t1-t0):.4f}s")

        # --- Optimized monitoring process ---
        C = update_C(C, Prob)
        t2 = time.perf_counter()
        print(f"C Update: {(t2-t1):.4f}s")

        # --- Entanglement Entropy calculation ---
        if jj in Steps1:
            S = entanglement_entropy_calc(C)
            data.append(np.real(S))
            t4 = time.perf_counter()
            print(f"EE calc.: {(t4-t2):.4f}s")
            
        t5 = time.perf_counter()

        print(f"whole run for t={jj}: {(t5-t0):.4f}s")
        print("__________________________________________")


    return data

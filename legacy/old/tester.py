import sys
import numpy as np
from scipy.stats import unitary_group
from scipy.linalg import expm, sinm, cosm
from scipy import linalg
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

def monitored_circuit(C, Prob):
    """
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
    return C


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
    pot1 = np.diag(pot1_diag)

    # Create diagonal matrices with phase and imaginary components
    pot_mat = np.diag(np.exp(1j * pot1_diag))

    K = np.dot(pot_mat, ham_exp)

    return K

L = 3

# Tight biding hamiltonian for a 1D lattice without potential and interactions
t_v = np.diag(np.ones(L-1),1)
t_v[L-1,0] = 1
t_v += t_v.T



for ii in range(1,3):
    mat = time_phase_disturbed_evol(t_v, ii)
    print(mat)
    print("_____________________")




"""
test = np.random.uniform(-1,1)

print(test)

# Energy eingenvalues and eigenvectores calculation
e1_t, v1_t = np.linalg.eigh(t_v)

ham_exp = linalg.expm(1j*t_v)

e_D = np.zeros([L,L], dtype=complex)
e_D += np.diag(np.exp(1j*e1_t))

v1_t_inv = linalg.inv(v1_t)

exp_ham = np.dot(e_D, v1_t_inv)
exp_ham = np.dot(v1_t, exp_ham) 

print(exp_ham)
print("_____________________")
print(ham_exp)
"""
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import sys
import time  
import ast

def get_simulation_params():
    arguments = sys.argv[1:]
    
    converted = []
    for item in arguments:
        try:
            converted.append(ast.literal_eval(item))
        except (ValueError, SyntaxError):
            converted.append(item)
    
    return converted

def redraw_status(status, printed_lines):
    """
    Redraw one line per PID in place.
    Returns the number of lines printed.
    """
    # status: Manager().dict {pid: t}
    items = sorted(status.items())  # stable order by pid

    # Move cursor back up to the start of previous block
    if printed_lines > 0:
        sys.stdout.write(f"\x1b[{printed_lines}A")

    lines = [f"[pid {pid}] t = {t}" for pid, t in items]

    for line in lines:
        sys.stdout.write("\x1b[2K")  # clear current line
        sys.stdout.write(line + "\n")

    sys.stdout.flush()
    return len(lines)

def list_organization(lst:list)->list:
    num_var = len(lst[0])
    out = [[] for _ in range(num_var)]

    for ii in range(len(lst)):
        for jj in range(num_var):
            out[jj].append(lst[ii][jj])

    return out


###############################################################################
# System Simulation utilities
###############################################################################

def hamiltonian_creator_2D(L: int, j: float = 1, periodic: bool = True) -> np.ndarray:
    """
    Creates a tight-binding Hamiltonian for a 2D square lattice with dimensions LxL.
    """
    dim = L * L
    h = np.zeros((dim, dim))

    # 1D chain hopping block
    mat_aux1 = np.diag(np.full(L - 1, -j), 1)
    mat_aux1 += np.diag(np.full(L - 1, -j), -1)

    if periodic:
        # horizontal boundary points
        mat_aux1[0, L - 1] = -j
        mat_aux1[L - 1, 0] = -j

        # vertical boundary points (wrap rows)
        h += np.diag(np.full(dim - 2 * L, -j),  2 * L)
        h += np.diag(np.full(dim - 2 * L, -j), -2 * L)

    # put 1D chains on the diagonal blocks
    for ii in range(L):
        h[ii * L:(ii + 1) * L, ii * L:(ii + 1) * L] = mat_aux1

    # couple neighbouring rows
    h += np.diag(np.full(dim - L, -j),  L)
    h += np.diag(np.full(dim - L, -j), -L)

    return h


def neel_state_creator_2D(L: int) -> np.ndarray:
    """
    Creates a 2D Néel state density matrix (occupation pattern alternating each row).
    """
    dim = L * L
    occ = np.zeros(dim, dtype=float)

    for idx in range(dim):
        row, col = divmod(idx, L)
        # 1 on "black" sublattice, 0 on "white" sublattice
        if (row + col) % 2 == 0:
            occ[idx] = 1.0

    # Density / correlation matrix is diagonal in site basis
    ns = np.diag(occ)
    return ns

def hamiltonian_creator_bethe(height: int, branching_factor: int = 3, j: float = 1, periodic: bool = True) -> np.ndarray:
    T = nx.balanced_tree(branching_factor, height)
    if periodic:
        node_tot = T.number_of_nodes()
        int_leaf = node_tot - 3**height
        leave_ind = [xx for xx in range(int_leaf, node_tot)]

        G2 = nx.random_regular_graph(branching_factor-1, 3**height)
        mapping = dict(zip(G2.nodes(), leave_ind))

        G3 = nx.relabel_nodes(G2, mapping)

        G_composed = nx.compose(T, G3)
    A_sparse = nx.to_scipy_sparse_array(G_composed)
    return A_sparse.todense()*j

def neel_state_creator_bethe(height: int) -> np.ndarray:
    lst = []
    for ii in range(height+1): 
        if ii%2==0: lst += [0]*(3**ii)
        if ii%2!=0: lst += [1]*(3**ii)
    return np.diag(lst)

def num_of_nodes(order:int)->int:
    if order==0: return 1
    res = 1
    for ii in range(order):
        res += 3*(2**ii)
    return res

def hamiltonian_creator_RRG(height: int, j: float = 1) -> np.ndarray:
    G1 = nx.random_regular_graph(3, num_of_nodes(height))
    A_sparse = nx.to_scipy_sparse_array(G1)
    return A_sparse.todense()*j

def neel_state_creator_RRG(height: int) -> np.ndarray:
    lst = []
    for ii in range(height+1): 
        if ii%2==0: lst += [0]*(3**ii)
        if ii%2!=0: lst += [1]*(3**ii)
    return np.diag(lst)

###############################################################################
# Quantities Calculation Funcs.
###############################################################################

def entropy(n):
    n_safe = np.clip(n, 1e-12, 1 - 1e-12)
    return -n_safe * np.log(n_safe) - (1 - n_safe) * np.log(1 - n_safe)

def mutual_info(C: np.ndarray, dim: str) -> float:
    L = C.shape[0]
    if dim=="1D": ref = L//2
    if dim=="2D": ref = L//2 - np.sqrt(L)//2
    pt = 0

    S_ref = entropy(C[ref, ref].real)

    for jj in range(L):
        S_jj = entropy(C[jj, jj].real)

        CA = C[np.ix_([ref, jj], [ref, jj])].astype(complex, copy=False)
        n = np.linalg.eigvalsh(CA).real
        S_joint = np.sum(entropy(n))

        pt += max(0.0, S_ref + S_jj - S_joint)

    return pt

def entanglement_entropy_half(C: np.ndarray, qt=False) -> float:
    """
    Von Neumann entanglement entropy of the left half of C.
    """
    
    L = C.shape[0]
    AAA1 = C[:L // 2, :L // 2]

    lamb = np.linalg.eigvalsh(AAA1)

    # S = - Tr[ rho log rho ] - Tr[ (1-rho) log (1-rho) ]
    lam_pos = lamb[lamb > 1e-20]
    S = -np.dot(lam_pos, np.log(lam_pos))

    lam_comp = 1 - lamb
    lam_comp_pos = lam_comp[lam_comp > 1e-20]
    S += -np.dot(lam_comp_pos, np.log(lam_comp_pos))

    return float(np.real(S))


def entanglement_entropy_quarter(C: np.ndarray) -> float:
    L = int(np.sqrt(C.shape[0]))

    C = np.asarray(C)
    
    idx0 = []
    for ii in range(L//2):
        idx0 += [x for x in range(2*ii*L//2, (2*ii+1)*L//2)]
    
    CA = C[np.ix_(idx0, idx0)].astype(complex, copy=False)
    n = np.linalg.eigvalsh(CA).real
    n = np.clip(n, 1e-12, 1 - 1e-12)

    return float(-np.sum(n*np.log(n) + (1-n)*np.log(1-n)))

def entanglement_entropy_bethe(C: np.ndarray, height:int) -> float:
    """
        Calculates the EEnt. for the interior of the Bethe lattice.
    """
    L = sum([3**ii for ii in range(height-1)])
    AAA1 = C[:L, :L]

    lamb = np.linalg.eigvalsh(AAA1)
    lam_pos = lamb[lamb > 1e-20]
    S = -np.dot(lam_pos, np.log(lam_pos))

    lam_comp = 1 - lamb
    lam_comp_pos = lam_comp[lam_comp > 1e-20]
    S += -np.dot(lam_comp_pos, np.log(lam_comp_pos))

    return float(S)

def IPR(C:np.ndarray, eps=1e-3):
    vals, vects = np.linalg.eigh(C)
    res = 0
    for ii in range(len(vals)):
        psi = vects[:, ii]          
        aux = np.abs(psi)**4         
        res += vals[ii] * np.sum(aux)

    return res

###############################################################################
# System Simulation
###############################################################################

def update_C(C: np.ndarray, Prob: float) -> np.ndarray:
    """
    Measurement/update channel on correlation matrix C.
    """
    L = C.shape[0]
    xx = np.random.random(L) > Prob         # True ~ keep / False ~ act
    C_diag = np.real(np.diag(C))

    for k in range(L):
        if not xx[k]:
            Prob1 = min(1.0, abs(C_diag[k]))
            meas_flip = (np.random.random() > Prob1)

            if not meas_flip:
                # Projective-like collapse to |1>
                A_ik = C[:, k]
                B_kj = C[k, :]
                denom = C[k, k]
                C_ikkj = np.outer(A_ik, B_kj) / denom
                C -= C_ikkj
                C[k, k] += 1.0
            else:
                # Collapse to |0>
                A_ik = -C[:, k].copy()
                A_ik[k] += 1.0
                B_kj = -C[k, :].copy()
                B_kj[k] += 1.0
                denom = 1.0 - C[k, k]
                C_ikkj = np.outer(A_ik, B_kj) / denom
                C += C_ikkj
                C[k, k] -= 1.0

    return C


def run_one_simulation_general(args):
    Prob, Time, Steps1, U0, C_init, EntEntr, Ballistic, amp, status, height = args

    # Make sure each worker has an independent RNG and state
    seed = os.getpid() ^ id(args)
    np.random.seed(seed % (2**32 - 1))

    K = U0
    C = C_init.copy()
    pid = os.getpid()
    data_EEnt = []
    data_IPR  = []
    data_MI  = []

    for jj in range(Time):

        if jj%10==0: 
            status[pid] = jj
        
        if jj!=0:
            C = K @ (C @ np.conjugate(K).T)
            C = update_C(C, Prob)

        if jj in Steps1: 
            # IPR:
            C1 = C + np.diag(np.random.uniform(-amp, amp, size=(len(C))))
            data_IPR.append(IPR(C1))

            # EEnt.:
            if EntEntr=="1|2" or EntEntr=="": S = entanglement_entropy_half(C)
            if EntEntr=="1|4": S = entanglement_entropy_quarter(C)
            if EntEntr=="bethe": S = entanglement_entropy_bethe(C, height)
            data_EEnt.append(S)

            # MI.:
            data_MI.append(mutual_info(C, "1D"))

    return data_IPR, data_EEnt, data_MI


def run_one_simulation_EEnt(args):
    Prob, Time, Steps1, U0, C_init, EntEntr, Ballistic, amp, status, height = args

    # Make sure each worker has an independent RNG and state
    seed = os.getpid() ^ id(args)
    np.random.seed(seed % (2**32 - 1))

    K = U0
    C = C_init.copy()
    pid = os.getpid()
    data_EEnt = []

    for jj in range(Time):

        if jj%10==0: 
            status[pid] = jj
        
        if jj!=0:
            C = K @ (C @ np.conjugate(K).T)
            C = update_C(C, Prob)

        if jj in Steps1: 
            #EEnt.:
            if EntEntr=="1|2" or EntEntr=="": S = entanglement_entropy_half(C)
            if EntEntr=="1|4": S = entanglement_entropy_quarter(C)
            if EntEntr=="bethe": S = entanglement_entropy_bethe(C, height)
            data_EEnt.append(S)

    return [], data_EEnt, []

def run_one_simulation_IPR(args):
    Prob, Time, Steps1, U0, C_init, EntEntr, Ballistic, amp, status, height = args

    # Make sure each worker has an independent RNG and state
    seed = os.getpid() ^ id(args)
    np.random.seed(seed % (2**32 - 1))

    K = U0
    C = C_init.copy()
    pid = os.getpid()
    data_IPR = []

    for jj in range(Time):

        if jj%10==0: 
            status[pid] = jj
        
        if jj!=0:
            C = K @ (C @ np.conjugate(K).T)
            C = update_C(C, Prob)

        if jj in Steps1: 
            #EEnt.:
            C1 = C + np.diag(np.random.uniform(-amp, amp, size=(len(C))))
            data_IPR.append(IPR(C1))

    return data_IPR, [], []


def run_one_simulation_MI(args):
    Prob, Time, Steps1, U0, C_init, EntEntr, Ballistic, amp, status, height = args

    # Make sure each worker has an independent RNG and state
    seed = os.getpid() ^ id(args)
    np.random.seed(seed % (2**32 - 1))

    K = U0
    C = C_init.copy()
    pid = os.getpid()
    data_MI = []

    for jj in range(Time):

        if jj%10==0: 
            status[pid] = jj
        
        if jj!=0:
            C = K @ (C @ np.conjugate(K).T)
            C = update_C(C, Prob)

        if jj in Steps1: 
            data_MI.append(mutual_info(C, "1D"))

    return [], [], data_MI


###############################################################################
# Utility to build the fixed unitary step and initial C for a given system size
###############################################################################

def build_initial_conditions(L:int, dim:str, res:float=1)->tuple:
    """
    Build:
    - U0: single-step unitary (Fourier-like hopping evolution)
    - C0: initial correlation matrix (Néel state here)
    """
    if dim=="1D":
        # Hamiltonian
        t_v = np.diag(np.ones(L - 1), 1)
        t_v[L - 1, 0] = 1
        t_v = t_v + t_v.T
        # Initial state: Néel pattern (alternating occupation)
        C0 = np.diag([i % 2 for i in range(L)]).astype(complex)

    if dim=="2D":
        # Hamiltonian
        t_v = hamiltonian_creator_2D(L)
        # Initial state
        C0 = neel_state_creator_2D(L)

    if dim=="bethe":
        # Hamiltonian
        t_v = hamiltonian_creator_bethe(L)
        # Initial state
        C0 = neel_state_creator_bethe(L)
    
    if dim=="RRG":
        # Hamiltonian
        t_v = hamiltonian_creator_RRG(L)
        # Initial state
        C0 = neel_state_creator_bethe(L)

    # U0 = exp(-i H dt). Here dt = 1.
    U0 = linalg.expm(-1j * t_v * res)
    print(U0.shape[0], C0.shape[0])
    return U0, C0


import cupy as cp
import cupyx.scipy.sparse as cps
import cupyx.scipy.sparse.linalg as cps_linalg
import os
import sys
import time

###############################################################################
# System Simulation utilities
###############################################################################
import cupy as cp
import cupyx.scipy.sparse as cps
import cupyx.scipy.sparse.linalg as cps_linalg

def hamiltonian_creator_2D_sparse(L: int, j: float = 1.0, periodic: bool = True):
    """
    Efficiently creates a 2D Hamiltonian using Sparse Matrices and Kronecker products.
    H_2D = (I_x ⊗ H_y) + (H_x ⊗ I_y)
    """
    # 1. Create a simple 1D hopping matrix (Tridiagonal)
    # Diagonals: -j on +1 and -1 off-diagonals
    ones = cp.ones(L - 1)
    off_diag = cp.full(L - 1, -j)
    
    # Base 1D Hamiltonian (Open Boundaries)
    h_1d = cps.diags([off_diag, off_diag], [1, -1], shape=(L, L))
    
    # Add Periodic Boundaries to 1D chain if requested
    if periodic:
        # Add corners: (0, L-1) and (L-1, 0)
        h_1d = h_1d + cps.diags([-j, -j], [L - 1, -(L - 1)], shape=(L, L))

    # 2. Build 2D Hamiltonian using Kronecker Product
    # Identity matrix of size L
    I = cps.eye(L)

    # H_2D = (H_1D ⊗ I) + (I ⊗ H_1D)
    # This automatically handles the row-wrapping logic perfectly.
    h_2d = cps.kron(h_1d, I) + cps.kron(I, h_1d)

    return h_2d

def neel_state_creator_2D_fast(L: int) -> cp.ndarray:
    """
    Creates a 2D Néel state density matrix using vectorized GPU operations.
    """
    dim = L * L

    idx = cp.arange(dim)

    rows = idx // L
    cols = idx % L

    mask = (rows + cols) % 2 == 0

    occ = mask.astype(float)

    return cp.diag(occ)


def entanglement_entropy_half(C: cp.ndarray, qt=False) -> float:
    L_total = C.shape[0]
    L_sub = L_total // 2
    C_sub = C[:L_sub, :L_sub]

    lamb = cp.linalg.eigvalsh(C_sub)
    lamb = cp.clip(lamb, 0.0, 1.0)

    valid_n = lamb > 1e-15
    S = -cp.dot(lamb[valid_n], cp.log(lamb[valid_n]))

    valid_1n = lamb < (1.0 - 1e-15)
    lam_comp = 1.0 - lamb[valid_1n]
    S -= cp.dot(lam_comp, cp.log(lam_comp))

    return float(S)

def entanglement_entropy_quarter(C: cp.ndarray) -> float:
    L_total = int(cp.sqrt(C.shape[0]))
    half_L = L_total // 2

    col_offsets = cp.arange(half_L)
    row_starts = cp.arange(half_L) * L_total
    
    idx_grid = row_starts[:, None] + col_offsets[None, :]
    idx0 = idx_grid.flatten()

    CA = C[cp.ix_(idx0, idx0)]

    n = cp.linalg.eigvalsh(CA)
    n = cp.clip(n, 1e-12, 1.0 - 1e-12)

    S = -cp.sum(n * cp.log(n) + (1 - n) * cp.log(1 - n))

    return float(S)

def IPR(C: cp.ndarray, eps=1e-3) -> float:
    vals, vects = cp.linalg.eigh(C)

    ipr_per_state = cp.sum(cp.abs(vects)**4, axis=0)
    
    res = cp.sum(vals * ipr_per_state)

    return float(res)

def update_C(C: cp.ndarray, Prob: float) -> cp.ndarray:
    
    L = C.shape[0]
    
    rand_checks = cp.random.random(L)
    measured_indices_gpu = cp.where(rand_checks < Prob)[0]
    
    if measured_indices_gpu.size == 0:
        return C

    measured_indices = measured_indices_gpu.get()

    for k in measured_indices:
        n_k = float(cp.real(C[k, k]))
        n_k = max(0.0, min(1.0, n_k))
        
        outcome_is_1 = cp.random.random() < n_k

        if outcome_is_1:
            denom = C[k, k] + 1e-18
            v_col = C[:, k]
            v_row = C[k, :]
            
            update_term = cp.outer(v_col, v_row) / denom
            C -= update_term
            C[k, k] = 1.0
            
        else:
            denom = 1.0 - C[k, k] + 1e-18
            
            v_col = -C[:, k]
            v_col[k] += 1.0
            
            v_row = -C[k, :]
            v_row[k] += 1.0
            
            update_term = cp.outer(v_col, v_row) / denom
            C += update_term
            C[k, k] = 0.0

    return C

def run_one_simulation_general(args):
    Prob, Time, Steps1, U0, C_init, EntEntr, Ballistic, amp, status = args

    seed = os.getpid() ^ id(args)
    cp.random.seed(seed % (2**32 - 1))

    K = cp.asarray(U0)
    C = cp.asarray(C_init)
    
    pid = os.getpid()
    data_EEnt = []
    data_IPR  = []

    for jj in range(Time):
        if status: # Safe check if status is a dummy dict or real manager
            status[pid] = jj
            
        if jj != 0:
            C = K @ C @ K.conj().T

        C = update_C(C, Prob)

        if jj in Steps1:
            # Generate noise directly on GPU
            noise = cp.random.uniform(-amp, amp, size=C.shape[0])
            C_noisy = C + cp.diag(noise)
            
            val_IPR = IPR(C_noisy)
            data_IPR.append(val_IPR)

            if EntEntr == "1|2" or EntEntr == "":
                val_S = entanglement_entropy_half(C)
            elif EntEntr == "1|4":
                val_S = entanglement_entropy_quarter(C)
            else:
                val_S = 0.0
            
            data_EEnt.append(val_S)

    return data_IPR, data_EEnt
import sys
import numpy as np
#from src.ent import *
from typing import Callable
#from src.reader_funcs import *
from pathlib import Path
import random


import numpy as np
import matplotlib.pyplot as plt

indices_to_mark = [3, 4, 6]

fig, ax = plt.subplots()

ax.imshow(data, cmap='Greys', vmin=0, vmax=1)

rows, cols = data.shape
for i in range(rows):
    for j in range(cols):
        current_index = i * cols + j
        
        if current_index in indices_to_mark:
            ax.text(j, i, str(current_index),
                    ha="center", va="center", color="red",
                    fontsize=5, fontweight='bold')


plt.show()

"""



L = [25, 50, 100, 250, 500, 1000]

pmax = 0.5
pstep = 0.05
ps = list(np.arange(0, pmax, pstep)) + list(np.arange(0.5, 1, 0.1))

for xx in L:
    dir_name = f"data/1D/syst_L={xx}/"
    fl_names = os.listdir(dir_name)
    for pp in fl_names:
        old = Path(dir_name+pp)

        new_name = old.name.replace(
           "(500,0.05,last(15))",
           "(500,1,last(15))"
        )

        old.rename(old.with_name(new_name))


lst_ind1 = [4]

print(random.sample(lst_ind1, 1))
lst_ind1 = [4,5,6,7,8,9,10,11,12]

lst_ind1 = [xx-4 for xx in lst_ind1]
lst_ind2 = lst_ind1.copy()

links = [[] for _ in range(len(lst_ind1))]

K = 3


for ii in lst_ind1:
    print(ii)
    if ii == -1: continue

    lst_ind2.remove(ii)
    el_num = len(links[ii])
    lst_aux = random.sample(lst_ind2, K-el_num)
    links[ii] += lst_aux.copy()


    for jj in lst_aux:
        links[jj].append(ii)
        if len(links[jj])==3: 
            lst_ind1[jj] = -1
            lst_ind2.remove(jj)

    lst_ind1[ii]=-1
    
    print(links)
    print(lst_ind2)
    print("_________________________________")

print(links)



L = [500]

pmax = 0.5
pstep = 0.05
ps = list(np.arange(0, pmax, pstep))

for xx in L:
    for pp in ps:
        old = Path(f"data/syst_L={xx}/IPR_{xx}_p={round(pp, 2)}_T=(500,0.05,last(15))_amo=0.005_Ballistic_1D")

        new_name = old.name.replace(
            "amo=0.005_",
            "amp=0.005_"
        )

        old.rename(old.with_name(new_name))

       
L = [12, 16, 20, 24]

pmax = 0.5
pstep = 0.05
ps = list(np.arange(0, pmax, pstep))

for xx in L:
    for pp in ps:
        old = Path(f"data/syst_L={xx}x{xx}_test/IPR_{xx*xx}_p={round(pp, 2)}_T=(500,0.05,last(15))_Ballistic_2D")

        new_name = old.name.replace(
            "_Ballistic_2D",
            "_amp=0.01_Ballistic_2D"
        )

        old.rename(old.with_name(new_name))




test = [([1,2], [11,1]), ([2,1], [1,11]), ([22,1], [11,11])]
out = list_organization(test)
EEqq_250_p=0.0_T=(500,0.05)_EntEntr=12_Ballistic_1D
print(out)

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
x = lambda ii: ii*100
print(type(x))
def cena(yy:int, lmbd:Callable[[int], int] = None)->int:
    if lmbd:
        return lmbd(yy)
     
    else: return 0

print(cena(10))



L = 12
C = neel_state_creator_2D(12)
ham = hamiltonian_creator_2D(12)

C1 = C + np.diag(np.random.uniform(-0.01, 0.01, size=(len(C))))
vals, vects = np.linalg.eigh(C1)
K = linalg.expm(-1j * ham)

print(IPR(C1))

for ii in range(100):
    C = K @ (C @ np.conjugate(K).T)

C1 = C + np.diag(np.random.uniform(-0.01, 0.01, size=(len(C))))
vals, vects = np.linalg.eigh(C1)
print(IPR(C1))

L = 16
idx0 = []
for ii in range(L//2):
    idx0 += [x for x in range(2*ii*L//2, (2*ii+1)*L//2)]

print(idx0)


# Build a labelled 16x16 matrix so we can see the selection clearly
L = 16
C = np.fromfunction(lambda i, j: 100*(i+1) + (j+1), (L, L), dtype=int)

# 1-based indices for the bottom-left quarter in your 4x4 labelling
idx_1based = [1, 2, 5, 6]

# --- Code under test (no complex cast) ---
C = np.asarray(C)
idx0 = np.asarray(idx_1based, dtype=int) - 1      # 1-based -> 0-based
CA = C[np.ix_(idx0, idx0)]                        # keep original dtype
# ------------------------------------------

print(C)
print("____________________________________________-")
print(CA)

print("dtype(C) =", C.dtype, "dtype(CA) =", CA.dtype)
print("idx0:", idx0.tolist())
print("CA:\n", CA)

# Expect rows/cols [0,1,4,5]
expected = np.array([
    [101, 102, 105, 106],
    [201, 202, 205, 206],
    [501, 502, 505, 506],
    [601, 602, 605, 606],
], dtype=C.dtype)

assert np.array_equal(CA, expected)
print("Selection OK.")


np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
L = 4
dim  = L*L

# 1) General-purpose matrix with negatives and zeros
C = np.zeros([dim,dim])
G = np.arange(-128, 128, dtype=float).reshape(dim, dim)

slice = [x+L for x in range(L//2)]



print(G[np.ix_(slice, slice)])


import os, multiprocessing as mp, mkl

def choose_PT(logical_cores):
    N_TRAJ = 30
    divisors = [30,15,10,6,5,3,2,1]
    P = next(d for d in divisors if d <= logical_cores)   # processes
    T = max(1, logical_cores // P)                        # MKL threads per proc
    return P, T

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    C = os.cpu_count()
    P, T = choose_PT(C)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["MKL_NUM_THREADS"] = str(T)
    os.environ["OMP_NUM_THREADS"]  = str(T)
    mkl.set_dynamic(False)
    mkl.set_num_threads(T)
    print(f"Using N_PROCESSES={P}, MKL threads per process={T} on {C} logical cores")
"""
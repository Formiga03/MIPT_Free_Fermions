from multiprocessing import Pool
from tqdm import tqdm
import sys
import os
from random import random
import numpy as np
from scipy.stats import unitary_group
from scipy import linalg
from tqdm import tqdm
from src.ent_og import *

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

# Time parameters
Time = 3000
t_step = 1
tt = list(range(0,Time,t_step))
Steps1 = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + \
         [Time-1]
Steps1 = tt

# Probability parameters
pmax = 0.4
pstep = 0.05
ps = np.arange(0, pmax+pstep, pstep)

# === Inside your main loop ===
for L in [8, 16, 32, 64]:
    print("__________________________________________________________________________")
    print("L=" + str(L))

    t_v = np.diag(np.ones(L - 1), 1)
    t_v[L - 1, 0] = 1
    t_v += t_v.T

    e1_t, v1_t = np.linalg.eigh(t_v)

    Data = []

    for Prob in ps:
        print("-p=" + str(Prob))

        # Prepare arguments for each parallel run
        args_list = [(L, Prob, Time, Steps1, t_v) for _ in range(30)]

        # Run in parallel
        with Pool(processes=15) as pool:
            data1 = list(tqdm(pool.imap(run_one_simulation_1D, args_list), total=30))

        # Save results
        np.savetxt(f"data/EEqq_{L}_p={Prob}_T=({Time},{t_step})", np.real(data1))
        Data.append(np.mean(data1, axis=0))

    np.savetxt(f"data/EEqq_{L}_p=({pmax},{pstep})_T=({Time},{t_step})_mean=30", np.real(Data))
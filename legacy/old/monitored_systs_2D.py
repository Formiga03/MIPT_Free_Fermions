"""
Optimaze code as much as possible, try to run a 50x50 and 100x100 simulation
"""
from multiprocessing import Pool
from tqdm import tqdm
import sys
import numpy as np
from scipy.stats import unitary_group
from scipy import linalg
from tqdm import tqdm
from legacy.ent import *

import mkl
mkl.set_num_threads( 16 )
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

exp = False
# Time parameters
Time = 1000
t_step = 1
tt = list(range(0,Time+t_step,t_step))
Steps1 = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + \
         [Time-1]


# Probability parameters
pmax = 1
pstep = 0.1
ps = np.arange(0, pmax+pstep, pstep)

for L in [10]:
    print("____________________________________________________________________________" 
          "_________________________")
    print("L="+str(L))

    dim = L*L

    t_v = hamiltonian_creator_2D(L)
    t_v = linalg.expm(-1j*t_v)

    C = neel_state_creator_2D(L)
    
    Data = []

    for Prob in ps: 
        print("-p=" + str(Prob))
        data =[]
        data1 = []

        # Prepare arguments for each parallel run
        args_list = [(dim, Prob, Time, Steps1, t_v, C) for _ in range(30)]

        # Run in parallel
        with Pool(processes=20) as pool:
            data1 = list(tqdm(pool.imap(run_one_simulation, args_list), total=30))

        # Save results
        np.savetxt(f"data/EEqq_{L}_p={Prob}_T=({Time},{t_step})_2D_exp", np.real(data1))
        Data.append(np.mean(data1, axis=0))
        
    #name = "data/EEqq_2D_"+str(L)+"x"+str(L)+"_p=("+str(pmax)+","+str(pstep)+")_T="+str(Time)
    #if exp: name += "-exp"
    #np.savetxt(name, np.real(Data))

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

# Time parameters
Time = 1000
t_step = 1
Steps1 = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + \
         [Time-1]
# Steps1 = list(range(0,Time,t_step))

# Probability parameters
pmax = 0.3
pstep = 0.05
ps = np.arange(0.3, pmax+pstep, pstep)


for L in [1000]:
    print("__________________________________________________________________________")
    print("L=" + str(L))

    dir_name = f"data/syst_L={L}"

    # hamiltonian generator
    t_v = np.diag(np.ones(L - 1), 1)
    t_v[L - 1, 0] = 1
    t_v += t_v.T

    t_v  = linalg.expm(-1j*t_v)

    # initial state generator
    C = np.zeros([L, L], dtype=complex)
    C = np.diag([x % 2 for x in range(L)])

    for Prob in ps:
        print("-p=" + str(Prob))

        # Prepare arguments for each parallel run
        args_list = [(Prob, Time, Steps1, t_v, C) for _ in range(1)]

        # Run in parallel
        with Pool(processes=10) as pool:
            data1 = list(tqdm(pool.imap(run_one_simulation, args_list), total=1))
        
        # Save results
        try:
            os.mkdir(dir_name)
            np.savetxt(f"{dir_name}/EEqq_{L}_p={Prob}_T=({Time},{t_step})_1D_test", np.real(data1))
        except FileExistsError:    
            np.savetxt(f"{dir_name}/EEqq_{L}_p={Prob}_T=({Time},{t_step})_1D_test", np.real(data1))



    
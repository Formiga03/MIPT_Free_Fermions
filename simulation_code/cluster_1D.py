import os
import sys
import numpy as np
import multiprocessing as mp
import time

from src.ent import *

N_MKL_THREADS = 1

os.environ["MKL_NUM_THREADS"] = str(N_MKL_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_MKL_THREADS)

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
# ============================================= Parameters ============================================= 
accumulate = True

Time = 500
t_step = 0.05
EEnt_num = 15
Tmode = f"last({EEnt_num})"

if "tot" in Tmode:  Steps1 = list(range(Time))
if "last" in Tmode: Steps1 = [0] + [Time - ii for ii in range(EEnt_num)]

pmax = 0.5
pstep = 0.05
ps = list(np.arange(0, pmax, pstep))

Ls = [500]        

EntEntr = "1|2"

ballistic = True
amp = 0.01

N_TRAJ = 10

N_PROCESSES = 10

def main():
    mp.set_start_method("spawn", force=True)

    for L in Ls:
        dir_name = f"data/syst_L={L}"
        os.makedirs(dir_name, exist_ok=True)

        U0, C0 = build_initial_conditions(L, "1D")

        for Prob in ps:
            args_list = [
                (round(Prob, 2), Time, Steps1, U0, C0, EntEntr, ballistic, amp, {})
                for _ in range(N_TRAJ)
            ]

            with mp.Pool(processes=N_PROCESSES) as pool:
                async_result = pool.map_async(run_one_simulation_general, args_list)
                
                async_result.wait()

                results_list = async_result.get()
                
            results_list = list_organization(results_list)

            out_arr_IPR  = np.array([np.real(r) for r in results_list[0]], dtype=object)
            out_arr_EEnt = np.array([np.real(r) for r in results_list[1]], dtype=object)

            base_name_EEnt = f"{dir_name}/EEqq_{L}_p={round(Prob, 2)}_T=({Time},{t_step},last({EEnt_num}))_EntEntr={EntEntr}"
            base_name_IPR = f"{dir_name}/IPR_{L}_p={round(Prob, 2)}_T=({Time},{t_step},last({EEnt_num}))_amp={amp}"

            if ballistic:
                base_name_EEnt += "_Ballistic"
                base_name_IPR  += "_Ballistic"

            file_EEnt = base_name_EEnt + "_1D.txt"
            file_IPR = base_name_IPR + "_1D.txt"

            if accumulate:
                with open(file_EEnt, "a") as f:
                    np.savetxt(f, out_arr_EEnt) 

                with open(file_IPR, "a") as f:
                    np.savetxt(f, out_arr_IPR) 

            else:
                np.savetxt(file_EEnt, out_arr_EEnt)
                np.savetxt(file_IPR, out_arr_IPR)

if __name__ == "__main__":
    main()
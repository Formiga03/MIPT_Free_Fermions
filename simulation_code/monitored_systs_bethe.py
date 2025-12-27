import os
import sys
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from src.ent import *

N_MKL_THREADS = 5

os.environ["MKL_NUM_THREADS"] = str(N_MKL_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_MKL_THREADS)

import mkl
mkl.set_num_threads(N_MKL_THREADS)
#print("MKL threads (parent):", mkl.get_max_threads())

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)


###############################################################################
# Simulation parameters
###############################################################################
accumulate = True
# Time Parameters
Time = 500
t_step = 0.05
EEnt_num = 15
Tmode = f"tot"

# Time instants for measurements
if "tot" in Tmode:  Steps1 = list(range(Time))
if "last" in Tmode: Steps1 = [0] + [Time - ii for ii in range(EEnt_num)]

# Measurement probability parameters
pmax = 1
pstep = 0.1
ps = list(np.arange(0.05, pmax, pstep))

# System sizes to scan
Ls = [2,3,4,5]         

# Entanglement Entropy Space
EntEntr = "bethe"

# Time Evolution Regime
ballistic = True

# Amplitude 
amp = 0.01

###############################################################################
# Driver parameters
###############################################################################
# How many parallel trajectories per (L, Prob)
N_TRAJ = 10

# How many OS processes to spawn
N_PROCESSES = 10

###############################################################################
# Main execution
###############################################################################
def main():
    # Very important for MKL + multiprocessing on Linux:
    mp.set_start_method("spawn", force=True)

    with mp.Manager() as manager:
        status = manager.dict()  # shared progress dict

        for L in Ls:
            print("================================================================")
            print(f"h = {L}")
            dir_name = f"data/syst_bethe_n={L}"
            os.makedirs(dir_name, exist_ok=True)

            # Build unitary step operator U0 and initial correlation matrix C0
            U0, C0 = build_initial_conditions(L, "bethe", t_step)

            for Prob in ps:
                print(f"- p = {round(Prob, 2)}")

                # clear status for this new p
                status.clear()

                # Prepare list of arguments, one per trajectory
                args_list = [
                    (round(Prob, 2), Time, Steps1, U0, C0, EntEntr, ballistic, amp, status, L)
                    for _ in range(N_TRAJ)
                ]

                with mp.Pool(processes=N_PROCESSES) as pool:
                    async_result = pool.map_async(run_one_simulation_general, args_list)

                    printed_lines = 0
                    start = time.perf_counter()

                    time.sleep(1)

                    # While trajectories are running, keep updating the table
                    while not async_result.ready():
                        if status:
                            printed_lines = redraw_status(status, printed_lines)
                        time.sleep(0.1)

                    # One last redraw at the end to get final values
                    if status:
                        printed_lines = redraw_status(status, printed_lines)

                    results_list = async_result.get()

                results_list = list_organization(results_list)

                elapsed = time.perf_counter() - start
                
                hours, rem = divmod(elapsed, 3600)
                minutes, seconds = divmod(rem, 60)
                print(f"-> Simulation Time: {int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}")

                # results_list is a list of lists (one list of entropies per traj)
                out_arr_IPR  = np.array([np.real(r) for r in results_list[0]], dtype=object)
                out_arr_EEnt = np.array([np.real(r) for r in results_list[1]], dtype=object)

                out_file_IPR = (
                    f"{dir_name}/IPR_{L*L}_p={round(Prob, 2)}_T=({Time},{t_step},{Tmode})_amp={amp}"
                )
                out_file_EEnt = (
                    f"{dir_name}/EEqq_{L*L}_p={round(Prob, 2)}_T=({Time},{t_step},{Tmode})_EntEntr={EntEntr}"
                )

                if ballistic:
                    out_file_EEnt += "_Ballistic"
                    out_file_IPR  += "_Ballistic"
                
                if accumulate:
                    with open(out_file_EEnt + "_2D", "a") as f:
                        np.savetxt(f,out_arr_EEnt) 

                    with open(out_file_IPR + "_2D", "a") as f:
                        np.savetxt(f,out_arr_IPR) 

                if not accumulate:
                    np.savetxt(out_file_EEnt + "_2D", out_arr_EEnt)
                    np.savetxt(out_file_IPR  + "_2D", out_arr_IPR)
                print(f"saved -> {out_file_EEnt}")
                print(f"saved -> {out_file_IPR}")



if __name__ == "__main__":
    main()

# print("jacques was here :)")


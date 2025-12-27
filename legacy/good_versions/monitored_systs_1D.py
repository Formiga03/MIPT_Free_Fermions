import os
import sys
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

from src.ent import *

N_MKL_THREADS = 16

os.environ["MKL_NUM_THREADS"] = str(N_MKL_THREADS)
os.environ["OMP_NUM_THREADS"] = str(N_MKL_THREADS)

import mkl
mkl.set_num_threads(N_MKL_THREADS)
#print("MKL threads (parent):", mkl.get_max_threads())

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

###############################################################################
# Simulation parameters
###############################################################################

# Time Parameters
Time = 500
t_step = 0.05

# Time instants for measurements
Steps1 = list(range(Time))
Steps1 = [Time - ii for ii in range(15)]

# Measurement probability parameters
pmax = 0.5
pstep = 0.05
ps = list(np.arange(0, pmax + pstep, pstep))

# System sizes to scan
Ls = [25, 50, 100, 250, 500]                                       

# Entanglement Entropy Space
EntEntr = ""

# Time Evolution Regime
ballistic = True  

###############################################################################
# Driver parameters
###############################################################################
# How many parallel trajectories per (L, Prob)
N_TRAJ = 7

# How many OS processes to spawn
N_PROCESSES = 7


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
            print(f"L = {L}")
            dir_name = f"data/syst_L={L}"
            os.makedirs(dir_name, exist_ok=True)

            # Build unitary step operator U0 and initial correlation matrix C0
            U0, C0 = build_initial_conditions(L, "1D")

            for Prob in ps:
                print(f"- p = {round(Prob, 2)}")

                # clear status for this new p
                status.clear()

                # Prepare list of arguments, one per trajectory
                args_list = [
                    (round(Prob, 2), Time, Steps1, U0, C0, EntEntr, ballistic, status)
                    for _ in range(N_TRAJ)
                ]

                with mp.Pool(processes=N_PROCESSES) as pool:
                    async_result = pool.map_async(run_one_simulation, args_list)

                    printed_lines = 0
                    start = time.perf_counter()

                    # While trajectories are running, keep updating the table
                    while not async_result.ready():
                        if status:
                            printed_lines = redraw_status(status, printed_lines)
                        time.sleep(0.1)

                    # One last redraw at the end to get final values
                    if status:
                        printed_lines = redraw_status(status, printed_lines)

                    results_list = async_result.get()

                elapsed = time.perf_counter() - start
                print(f"-> Simulation Time: {elapsed:.1f} s")

                # results_list is a list of lists (one list of entropies per traj)
                out_arr = np.array([np.real(r) for r in results_list], dtype=object)

                out_file = (
                    f"{dir_name}/EEqq_{L}_p={round(Prob, 2)}_T=({Time},{t_step})_EntEntr={EntEntr}"
                )
                if ballistic:
                    out_file += "_Ballistic"
                np.savetxt(out_file + "_1D", out_arr)
                print(f"saved -> {out_file}")


if __name__ == "__main__":
    main()

"""
def main():
    # Very important for MKL + multiprocessing on Linux:
    mp.set_start_method("spawn", force=True)

    for L in Ls:
        print("================================================================")
        print(f"L = {L}")
        dir_name = f"data/syst_L={L}"
        os.makedirs(dir_name, exist_ok=True)

        # Build unitary step operator U0 and initial correlation matrix C0
        U0, C0 = build_initial_conditions(L, "2D", t_step)

        for Prob in ps:
            print(f"- p = {Prob}")

            # Prepare list of arguments, one per trajectory
            args_list = [
                (Prob, Time, Steps1, U0, C0, "")
                for _ in range(N_TRAJ)
            ]

            # Run trajectories in parallel
            with mp.Pool(processes=N_PROCESSES) as pool:
                # pool.imap returns generator, tqdm wraps it nicely
                results_list = list(tqdm(pool.imap(run_one_simulation, args_list),
                                         total=N_TRAJ))

            # results_list is a list of lists (one list of entropies per traj)
            # Stack or average as needed; here we just save raw
            out_arr = np.array([np.real(r) for r in results_list], dtype=object)

            out_file = (
                f"{dir_name}/EEqq_{L}_p={Prob}_T=({Time},{t_step})_1D"
            )
            np.savetxt(out_file, out_arr)
            print(f"saved -> {out_file}")
"""
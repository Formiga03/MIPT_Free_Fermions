import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True


sims = ["1D", "2D", "2D", "2D", "2D", "2D", "2D"]

sizes = [[25, 50, 100, 250, 500],
         [12, 16, 20],
         [12, 16, 20],
         [12, 16, 20],
         [12, 16, 20],
         [12, 16, 20],
         [12, 16, 20]]

kws = [["EEqq", "(500,1,last(15))", "Ballistic"],
       ["EEqq", "(500,1,last(15))", "Ballistic"],
       ["EEqq", "(500,0.05,last(30))", "Ballistic"],
       ["EEqq", "(500,0.1,last(30))", "Ballistic"],
       ["EEqq", "(500,0.25,last(30))", "Ballistic"],
       ["EEqq", "(500,0.5,last(30))", "Ballistic"],
       ["EEqq", "(500,0.75,last(30))", "Ballistic"]
       ]

for dm in [1, 2, 3, 4, 5, 6]:
    T_S_plot = False
    P_S_plot = True
    sizes1 = sizes[dm]

    Data1 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
               #  (MP:list[float], s_aveg:list[float]):L2, ...]

    print(f"-{sims[dm]}:")
    for L in tqdm(sizes1):

        dt8 = raw_data_reader(L, sims[dm],  kws[dm])
        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation
        print(len(dt8[0][0]))
        MP = [kk[0] for kk in dt8[1]] 
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 

        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                
                tt = list(range(500))
                # Average
                fig, axs = plt.subplots(1, 1, figsize=(10, 5))
                plt.plot(tt, dt_ave[ii], "o")
                plt.xlabel("time")
                plt.ylabel(r'E[S]')
                plt.title(f'Entanglement Entropy vs. Time {dt8[1][ii][-1]}D, Num. of Iters={len(dt8[0][ii])}, MP={str(MP[ii])}')
                plt.savefig(f"plots/EEvsT=({dt8[1][0][1]},{dt8[1][0][2]})_L={L}_Iters={len(dt8[0][ii])}_MP={MP[ii]}_{dt8[1][ii][-1]}D.png")
                plt.show()
                plt.close()
        
        plt.clf()
        
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data1.append((MP, s_aveg))

    if P_S_plot:
        for jj in range(len(sizes1)):
            plt.plot(Data1[jj][0], Data1[jj][1], "-o", label=f"L={sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Averaged Entanglement\nEntropy (Time Step = {dt8[1][0][2]}) {dt8[1][0][-1]}D - Ballistic Regime")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

"""
    # Linear Case:
        for jj in range(len(sizes1)):
            var1 = [Data1[jj][1][ii]/np.log(sizes1[jj]) for ii in range(len(Data1[jj][0]))]
            
            plt.plot(Data1[jj][0], var1, "-o", label=f"L={sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]/L")

        plt.legend()
        plt.title(f"Measurement Probability vs. Linear Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime")
        plt.savefig(f"plots/MPvsS_Ave(S|L)_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

        # Linear Case:
        for jj in range(len(sizes1)-1):
            var1 = [(Data1[jj+1][1][ii] - Data1[jj][1][ii])/(sizes1[jj+1] - sizes1[jj])
                   for ii in range(len(Data1[jj][0]))]
            
            plt.plot(Data1[jj][0], var1, "-o", label=f"L={sizes1[jj+1] - sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("dE[S]/dL")

        plt.legend()
        plt.title(f"Measurement Probability vs. Linear Variations of Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime")
        plt.savefig(f"plots/MPvsVar_S_Ave(dS\dL)_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

        # Log Case:
        for jj in range(len(sizes1)-2):
            var1 = [(Data1[jj+1][1][ii] - Data1[jj][1][ii])/(np.log(sizes1[jj+1]) - np.log(sizes1[jj]))
                   for ii in range(len(Data1[jj][0]))]
            
            plt.plot(Data1[jj][0], var1, "-o", label=f"L={sizes1[jj+1] - sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("dE[S]/dlog(L)")

        plt.legend()
        plt.title(f"Measurement Probability vs. Log. Variations of Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime")
        plt.savefig(f"plots/MPvsVar_S_Ave(dS\dlog(L))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()
"""
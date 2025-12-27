import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True


T_S_plot = False
Var_analysis = False
Var_MP_plot = False
P_S_plot = True

# ======================================== 1D ========================================
# 1D Case Parameters
dim1 = False
sizes1 = [25, 50, 100, 250, 500]

Data1 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
           #  (MP:list[float], s_aveg:list[float]):L2, ...]

Var_Data1 = [] # [[(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L1, 
               #  [(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L2, ...]

if dim1:
    print("-1D:")
    for L in tqdm(sizes1):

        lmbd_func = lambda x: -np.log(2*x/L)
        dt8 = raw_data_reader(L, "1D", ["IPR", "0.05", "Ballistic", "amp=0.01"], lmbd_func)

        # Averaging over the trajectories
        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation

        MP = [kk[0] for kk in dt8[1]] 
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 

        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                #tt = np.linspace(0, Time[ii][0], Time[ii][0]//Time[ii][1])
                #tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(1000),0.15)] + [1000-1]
                #print(tt)

                tt = list(range(500))
                # Average
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].plot(tt, dt_ave[ii])
                axs[0].set_xlabel("time")
                axs[0].set_ylabel(r'E[S]')
                axs[0].set_title("Mean")

                # Standard Diveation
                axs[1].plot(tt, dt_std[ii])
                axs[1].set_xlabel("time")
                axs[1].set_ylabel(r'$\sigma[S]$')
                axs[1].set_title("Standard Deviation")

                fig.suptitle(f'Entanglement Entropy vs. Time {dt8[1][ii][3]}D, Num. of Iters={len(dt8[0][ii])}, MP={str(MP[ii])}')
                plt.savefig(f"plots/EEvsT_L={L}_Iters={len(dt8[0][ii])}_MP={MP[ii]}_{dt8[1][ii][3]}D.png")
                #plt.show()
                plt.close()
        
        plt.clf()

        # Averaging the plataue
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data1.append((MP, s_aveg))

    if P_S_plot:
#       ======================================== MP Vs. IPR Plot ========================================
        for jj in range(len(sizes1)):
            plt.plot(Data1[jj][0], Data1[jj][1], "-o", label=f"L={sizes1[jj]}")
            plt.xlabel("MP")
            plt.ylabel("E[-log(IPR)]")

        plt.legend()
        #plt.semilogy()
        plt.title(f"Measurement Probability vs. Averaged IPR {dt8[1][0][-1]}D\n- Ballistic Regime")
        plt.savefig(f"plots/MPvsIPR_Ave_L={sizes1}_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

# ======================================== 2D ========================================

# 2D Case Parameters
dim2 = True

sizes2 = [12, 16, 20, 24, 28]

Data2 = []     # [(MP:list[float], s_aveg:list[float]):L1, 
               #  (MP:list[float], s_aveg:list[float]):L2, ...]

Var_Data2 = [] # [[(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L1, 
               #  [(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L2, ...]


if dim2:
    print("-2D:")
    for L in tqdm(sizes2):

        amp = 0.01
        lmbd_func = lambda x: -np.log(x/(L*L))
        dt8 = raw_data_reader(L, "2D", ["IPR", f"amp={amp}", "Ballistic"], lmbd_func)

        # Averaging over the trajectories
        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation

        MP = [kk[0] for kk in dt8[1]] 
        print(MP)
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 

        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                #tt = np.linspace(0, Time[ii][0], Time[ii][0]//Time[ii][1])
                #tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(1000),0.15)] + [1000-1]
                #print(tt)
                tt = list(np.arange(Time-15, Time, 1))
                #tt = list(range(500))

                # Average
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].plot(tt, dt_ave[ii])
                axs[0].set_xlabel("time")
                axs[0].set_ylabel(r'E[S]')
                axs[0].set_title("Mean")

                # Standard Diveation
                axs[1].plot(tt, dt_std[ii])
                axs[1].set_xlabel("time")
                axs[1].set_ylabel(r'$\sigma[S]$')
                axs[1].set_title("Standard Deviation")

                fig.suptitle(f'IPR vs. Time {dt8[1][ii][3]}D, Num. of Iters={len(dt8[0][ii])}, MP={str(MP[ii])}')
                plt.savefig(f"plots/IPRvsT_L={L}x{L}_Iters={len(dt8[0][ii])}_MP={MP[ii]}_{dt8[1][ii][3]}D.png")
                #plt.show()
                plt.close()
        
        plt.clf()
        
        # Averaging the plataue
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data2.append((MP, s_aveg))

    if P_S_plot:
#       ======================================== MP Vs. IPR Plot ========================================
        for jj in range(len(sizes2)):
            var1 = [Data2[jj][1][ii]/(L*L) for ii in range(len(Data2[jj][0]))]

            plt.plot(Data2[jj][0], var1, "-o", label=f"L={sizes2[jj]}")
            plt.xlabel("MP")
            plt.ylabel("E[-log(IPR)]")

        #plt.semilogy()
        plt.legend()
        plt.title(f"Measurement Probability vs. Averaged IPR {dt8[1][0][-1]}D\n- Ballistic Regime")
        plt.savefig(f"plots/MPvs-log(IPR)_ave_L={sizes2}_Ballistic_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

#       ======================================== MP Vs. EEnt. Plot ========================================
        for jj in range(len(sizes2)-1):
        
            var1 = [(Data2[jj+1][1][ii] - Data2[jj][1][ii])/(np.log(sizes2[jj+1]**2) - np.log(sizes2[jj]**2))
                   for ii in range(len(Data2[jj][0])-1)]
            
            print(var1)
            print("__________________________________________")
            print([(Data2[jj+1][1][ii] - Data2[jj][1][ii]) for ii in range(len(Data2[jj][0]))])
            print("__________________________________________")
            print(np.log(sizes2[jj+1]**2) - np.log(sizes2[jj]**2))
            print("__________________________________________")
            print("__________________________________________")

            plt.plot(Data2[jj][0], var1, "-o", 
                     label=f"L={sizes2[jj+1]}-{sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[-Log(IPR)]")
         
        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of IPR \n{dt8[1][0][-1]}D - Ballistic Regime")
        plt.savefig(f"plots/MPvsIPR_var_L={sizes2}_{dt8[1][0][-1]}D_Ballistic_log.png")
        plt.show()
        plt.close()

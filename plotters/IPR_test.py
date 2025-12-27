import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True


# ======================================== 1D ========================================

# 1D Case Parameters
dim1 = False
sizes1 = [25, 50, 100, 250, 500]

Data1 = []     # [(MP:list[float], s_aveg:list[float]):Amp1, 
               #  (MP:list[float], s_aveg:list[float]):Amp2, ...]

amps = [0.01, 0.02]

lmbd_func = lambda x: -np.log(2*x/(L))
if dim1:
    print("-1D:")
    for L in sizes1:
        for aa in amps:
            dt8 = raw_data_reader(L, "1D", ["IPR", f"_amp={aa}_", "Ballistic"], lmbd_func)
            
            # Averaging over the trajectories
            dt_ave = trajectory_average(dt8[0]) # Mean
            MP = [kk[0] for kk in dt8[1]] 

            # Averaging the plataue
            limit = 1
            s_aveg = s_average(dt_ave, limit)
            Data1.append((MP, s_aveg))

        #======================================== MP Vs. IPR Plot ========================================
        for jj in range(len(amps)):
            plt.plot(Data1[jj][0], Data1[jj][1], "-o", label=f"Amp={amps[jj]}")


        plt.xlabel("MP")
        plt.ylabel("-log(IPR)")
        #plt.semilogy()
        plt.title(f"Measurement Probability vs IPR Plot Amplitude Dependence Test\nfor Lattice Size {L} - Ballistic Regime")
        plt.legend()
        plt.savefig(f"plots/MPvsIPR_L={L}_amps={amps}_dependence_test.png")
        #plt.show()
        plt.close()

        Data1.clear()


# ======================================== 2D ========================================

# 2D Case Parameters
dim2 = True

sizes2 = [12, 16, 20, 24]

Data2 = []     # [(MP:list[float], s_aveg:list[float]):Amp1, 
               #  (MP:list[float], s_aveg:list[float]):Amp2, ...]

amps = [0.005, 0.01, 0.02]

lmbd_func = lambda x: -np.log(x/(L*L))

print("-2D:")
for L in sizes2:
    for aa in amps:
        dt8 = raw_data_reader(L, "2D", ["IPR", f"_amp={aa}_", "Ballistic"], lmbd_func)
        
        # Averaging over the trajectories
        dt_ave = trajectory_average(dt8[0]) # Mean
        MP = [kk[0] for kk in dt8[1]] 

        # Averaging the plataue
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data2.append((MP, s_aveg))

    #======================================== MP Vs. IPR Plot ========================================
    for jj in range(len(amps)):
        plt.plot(Data2[jj][0][:10], Data2[jj][1][:10], "-o", label=f"Amp={amps[jj]}")


    plt.xlabel("MP")
    plt.ylabel("-log(IPR)")
    #plt.semilogy()
    plt.title(f"Measurement Probability vs IPR Plot Amplitude Dependence Test\nfor Lattice Size {L}x{L} - Ballistic Regime")
    plt.legend()
    plt.savefig(f"plots/MPvsIPR_L={L}_amps={amps}_dependence_test.png")
    plt.show()
    plt.close()

    Data2.clear()
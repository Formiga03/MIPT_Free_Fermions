import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True

sims = ["1D","2D"]

sizes = [[25, 50, 100, 250, 500],
        [12, 16, 20, 24, 28, 32, 36]]

kws = [ ["IPR", "(500,1,last(30))", "Ballistic"],
        ["IPR", "(500,0.05,last(30))", "Ballistic"]]
Nkws = [ ["p=0.6","p=0.7","p=0.8","p=0.9"],
        ["p=0.6","p=0.7","p=0.8","p=0.9"]]
lmbd_func = lambda x: -np.log(2*x/(L))
for dm in [0]:
    T_S_plot = False
    P_S_plot = True
    sizes1 = sizes[dm]

    Data1 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
               #  (MP:list[float], s_aveg:list[float]):L2, ...]

    print(f"-{sims[dm]}:")
    for L in tqdm(sizes1):

        dt8 = raw_data_reader(L, sims[dm],  kws[dm], Nkws[dm], lmbd_func)
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

        plot_title = f"Measurement Probability vs. Averaged IPR\n{dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsIPR_Ave(E[-log(IPR)]))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "E[-log(IPR)]"
        plot_mp_vs_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel)
    
        Dt = invert_data_structure(Data1.copy(), sizes1)
        
        plot_title = f"System Size vs. Averaged IPR\n{dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/LvsIPR_Ave(E[-log(IPR)]))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        xlabel     = "L"
        ylabel     = "E[-log(IPR)]"
        plot_mp_vs_quant(Dt[0], Dt[1], plot_title, filename, xlabel=xlabel, ylabel=ylabel)
        
        # Linear Case:
        perimeter  = lambda xx: 2*xx-4
        plot_title = f"Measurement Probability vs. Log. Variations of Averaged IPR\n{dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsVar_IPR_Ave(dE[-log(IPR)])\dlog(L))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "dE[-log(IPR)]/dL"
        plot_mp_vs_var_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, func=perimeter)

        # Quadratic Case:
        area = lambda xx: xx*xx/4
        plot_title = f"Measurement Probability vs. Quadratic Variations of Averaged IPR\n{dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsVar_IPR_Ave(dE[-log(IPR)])\dL**2)_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "dE[-log(IPR)]/dL**2"
        plot_mp_vs_var_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, func=area)
        
       

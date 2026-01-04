import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True

sims = ["1D","2D"]

sizes = [[25, 50, 100, 250, 500, 1000],
        [12, 16, 20, 24, 28, 32]]

kws = [ ["EEqq", "(500,1,last(30))", "Ballistic"],
        ["EEqq", "(500,0.05,last(30))", "Ballistic"]]

Nkws = [["p=0.6","p=0.7","p=0.8","p=0.9", "p=0.0_"],
        ["p=0.6","p=0.7","p=0.8","p=0.9", "p=0.0_"]]

for dm in [0]:
    T_S_plot = False
    P_S_plot = True
    sizes1 = sizes[dm]

    Data1 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
               #  (MP:list[float], s_aveg:list[float]):L2, ...]

    print(f"-{sims[dm]}:")
    for L in tqdm(sizes1):

        dt8 = raw_data_reader(L, sims[dm],  kws[dm], Nkws[dm])
        dt_ave = trajectory_average(dt8[0]) # Mean

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

        plot_title = f"Measurement Probability vs. Averaged Entanglement\nEntropy (Time Step = {dt8[1][0][2]}) {dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsS_Ave_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "E[S]"
        plot_mp_vs_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, xlimits=(0,0.5))
        
        func1  = lambda xx1, xx2: xx1/xx2
        plot_title = f"Measurement Probability vs. Averaged Entanglement\nEntropy (Time Step = {dt8[1][0][2]}) {dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsS_Ave(S|L)_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "E[S]/L"
        plot_mp_vs_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, xlimits=(0,0.5), func_size1=func1)


        Dt = invert_data_structure(Data1.copy(), sizes1)
        
        plot_title = f"System Size vs. Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/LvsS_Ave_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        xlabel     = "L"
        ylabel     = "E[S]"
        plot_mp_vs_quant(Dt[0], Dt[1], plot_title, filename, xlabel=xlabel, ylabel=ylabel)

        # Linear Case:
        perimeter  = lambda xx: 2*xx-4
        plot_title = f"Measurement Probability vs. Linear Variations of Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime"
        filename   = f"plots/MPvsVar_S_Ave(dS\dL)_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
        ylabel     = "dE[S]/dL"
        plot_mp_vs_var_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, func=perimeter, xlimits=(0,0.5))

        if sims[dm]=="1D":
            # Log Case:
            area = lambda xx: np.log(xx)
            plot_title = f"Measurement Probability vs. Log. Variations of Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime"
            filename   = f"plots/MPvsVar_S_Ave(dS\dlog(L))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
            ylabel     = "dE[S]/dlog(L)"
            plot_mp_vs_var_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, func=area, xlimits=(0,0.5))
        if sims[dm]=="2D":
            # Quadratic Case:
            area = lambda xx: xx*xx/4
            plot_title = f"Measurement Probability vs. Quadratic Variations of Averaged Entanglement\nEntropy {dt8[1][0][-1]}D - Ballistic Regime"
            filename   = f"plots/MPvsVar_S_Ave(dS\dL**2))_L={sizes1}_T=({dt8[1][0][1]},{dt8[1][0][2]})_Ballistic_{dt8[1][0][-1]}D.png"
            ylabel     = "dE[S]/dL**2"
            plot_mp_vs_var_quant(Data1, sizes1, plot_title, filename, ylabel=ylabel, func=area, xlimits=(0,0.5))

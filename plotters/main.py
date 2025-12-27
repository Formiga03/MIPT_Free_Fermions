import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.reader_funcs import *
#plt.rcParams['text.usetex'] = True

T_S_plot = True
Var_analysis = False
Var_MP_plot = False
P_S_plot = True

#################################### 1D ################################################
# 1D Case Parameters
dim1 = False
sizes1 = [25, 50, 100, 250]
Data1 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
           #  (MP:list[float], s_aveg:list[float]):L2, ...]

Var_Data1 = [] # [[(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L1, 
               #  [(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L2, ...]

if dim1:
    print("-1D:")
    for L in tqdm(sizes1):

        dt8 = raw_data_reader(L, "1D", ["EEqq", "last(15)", "1|2_1D"])
        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_ave = np.average(dt8[0], axis=1)

        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation

        MP = [kk[0] for kk in dt8[1]] 
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 

        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                #tt = np.linspace(0, Time[ii][0], Time[ii][0]//Time[ii][1])
                
                #tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(1000),0.15)] + [1000-1]
                #print(tt)
                tt = list(np.arange(Time-15, Time, 1))
                print(tt)

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
        
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data1.append((MP, s_aveg))

        if Var_analysis:
            res = []

            for ii in range(len(dt8[0])):

                lst_aux1 = [x1[limit:] for x1 in dt8[0][ii]]
                lst_aux2 = [[x2*x2 for x2 in iter] for iter in lst_aux1]

                var = np.var(lst_aux1, axis=1)
                s_p2 = np.mean(lst_aux2, axis=1)
                rel_flut = [var[jj]/s_p2[jj] for jj in range(len(var))]

                var_ave = np.mean(var)
                RF_ave = np.mean(rel_flut)

                res.append((MP[ii], var_ave, RF_ave))

            Var_Data1.append(res)

    if Var_MP_plot:

        for jj in range(len(sizes1)):

            var_avg = [x[1] for x in Var_Data1[jj]] 
            var_RF = [x[2] for x in Var_Data1[jj]] 
            MP = [x[0] for x in Var_Data1[jj]] 

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].plot(MP, var_avg, "-o", label=f"L={sizes1[jj]}")
            axs[0].set_xlabel("MP")
            axs[0].set_ylabel('E[Var[S]]')
            axs[0].set_title("Average Variance")

            # Standard Diveation
            axs[1].plot(MP, var_RF, "-o", label=f"L={sizes1[jj]}")
            axs[1].set_xlabel("MP")
            axs[1].set_ylabel('E[Var[S]]/E[S2]')
            axs[1].set_title("Relative Fluctuation")

            fig.suptitle(f'Variance Analysis 1D, L = {sizes1[jj]}, Num. of Iters={30}')
            plt.savefig(f"plots/VarvsMP_L={sizes1[jj]}_1D.png")
            #plt.show()
            plt.close()

            plt.plot(Var_Data1[jj][0], Var_Data1[jj][1], "-o", label=f"L={sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

    if P_S_plot:

        for jj in range(len(sizes1)):
            plt.plot(Data1[jj][0], Data1[jj][1], "-o", label=f"L={sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        print(dt8[1])
        plt.title(f"Measurement Probability vs. Averaged Entanglement Entropy\n{dt8[1][0][-1]}D - Random Phase Regime")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes1}_randph_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()

        # Linear case
        for jj in range(len(sizes1)-1):
        
            var1 = [(Data1[jj+1][1][ii] - Data1[jj][1][ii])/(sizes1[jj+1] - sizes1[jj])
                   for ii in range(len(Data1[jj][0]))]
            
            plt.plot(Data1[jj][0], var1, "-o", 
                     label=f"L={sizes1[jj+1]}-{sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        
        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy\n{dt8[1][0][-1]}D Linear Case - Random Phase Regime")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes1}_randph_{dt8[1][0][-1]}D_Linear.png")
        plt.show()
        plt.close()

        # Log case
        for jj in range(len(sizes1)-1):

            var1 = [(Data1[jj+1][1][ii] - Data1[jj][1][ii])/
                   (np.log(sizes1[jj+1]) - np.log(sizes1[jj])) 
                   for ii in range(len(Data1[jj][0]))]

            plt.plot(Data1[jj][0], var1, "-o", label=f"L={sizes1[jj+1]}-{sizes1[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.semilogx()
        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy\n{dt8[1][0][-1]}D Logarithmic Case - Random Phase Regime")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes1}_randph_{dt8[1][0][-1]}D_Log.png")
        plt.show()
        plt.close()

#################################### 2D ################################################
# 2D Case Parameters
dim2 = False
sizes2 = [12, 16, 20]

Data2 = [] # [(MP:list[float], s_aveg:list[float]):L1, 
           #  (MP:list[float], s_aveg:list[float]):L2, ...]

Var_Data2 = [] # [[(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L1, 
               #  [(MP1, Var1, Rel_Flut1),(MP2, Var2, Rel_Flut2), ...]:L2, ...]

MP_Data2 = []

if dim2:
    print("-2D:")
    for L in tqdm(sizes2):

        dt8 = raw_data_reader(L, "2D", ["0.05", "500", "EEqq", "1|4_2D"])

        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation

        MP = [kk[0] for kk in dt8[1]] 
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 

        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                #tt = np.linspace(0, Time[ii][0], Time[ii][0]//Time[ii][1])
                #tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time[ii][0]),0.15)] + \
                #      [Time[ii][0]-1]
                tt = [500 - ii for ii in range(15)]

                #tt = list(range(500))

                # Average
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].plot(tt, dt_ave[ii])
                axs[0].set_xlabel("time")
                axs[0].set_ylabel(r'E[S]')
                axs[0].set_title("Mean")

                # Standard Diveation
                #axs[1].plot(tt, dt_std[ii])
                axs[1].set_xlabel("time")
                axs[1].set_ylabel(r'$\sigma[S]$')
                axs[1].set_title("Standard Deviation")

                fig.suptitle(f'Entanglement Entropy vs. Time {dt8[1][ii][3]}D, Num. of Iters={len(dt8[0][ii])}, MP={format(MP[ii], ".2f")}')
                plt.savefig(f"plots/EEvsT_L={L}x{L}_Iters={len(dt8[0][ii])}_MP={format(MP[ii], ".2f")}_{dt8[1][ii][3]}D_exp.png")
                plt.show()
                plt.close()
        
        plt.clf()
        
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Data2.append((MP, s_aveg))

        if Var_analysis:

            res = []

            for ii in range(len(dt8[0])):

                lst_aux1 = [x1[18:] for x1 in dt8[0][ii]]
                lst_aux2 = [[x2*x2 for x2 in iter] for iter in lst_aux1]

                var = np.var(lst_aux1, axis=1)
                s_p2 = np.mean(lst_aux2, axis=1)
                rel_flut = [var[jj]/s_p2[jj] for jj in range(len(var))]

                var_ave = np.mean(var)
                RF_ave = np.mean(rel_flut)

                res.append((MP[ii], var_ave, RF_ave))

            Var_Data2.append(res)
            print(f"L={L}: {Var_Data2}")

    if Var_MP_plot:

        for jj in range(len(sizes2)):

            var_avg = [x[1] for x in Var_Data2[jj]] 
            var_RF = [x[2] for x in Var_Data2[jj]] 
            MP = [x[0] for x in Var_Data2[jj]] 

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].plot(MP, var_avg, "-o", label=f"L={sizes2[jj]}")
            axs[0].set_xlabel("MP")
            axs[0].set_ylabel('E[Var[S]]')
            axs[0].set_title("Average Variance")

            # Standard Diveation
            axs[1].plot(MP, var_RF, "-o", label=f"L={sizes2[jj]}")
            axs[1].set_xlabel("MP")
            axs[1].set_ylabel('E[Var[S]]/E[S2]')
            axs[1].set_title("Relative Fluctuation")

            fig.suptitle(f'Variance Analysis 1D, L = {sizes2[jj]}, Num. of Iters={5}')
            plt.savefig(f"plots/VarvsMP_L={sizes2[jj]}_2D.png")
            #plt.show()
            plt.close()

            plt.plot(Var_Data2[jj][0], Var_Data2[jj][1], "-o", label=f"L={sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

    if P_S_plot:
        
#       ======================================== P Vs. EEnt. Plot ========================================
        for jj in range(len(sizes2)):
            plt.plot(Data2[jj][0], Data2[jj][1], "-o", label=f"L={sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Averaged Entanglement Entropy\n{dt8[1][0][-1]}D - Random Phase Regime")
        #plt.savefig(f"plots/MPvsS_Ave_L={sizes2}_randph_{dt8[1][0][-1]}D.png")
        #plt.show()
        plt.close()

         # Log case
        for jj in range(2):

            var1 = [(Data2[jj+1][1][ii] - Data2[jj][1][ii])/(sizes2[jj+1]- sizes2[jj])
                   for ii in range(15)]

            print(var1)
            print("__________________________________________")
            print([(Data2[jj+1][1][ii] - Data2[jj][1][ii])
                   for ii in range(len(Data2[jj][0]))])
            print("__________________________________________")
            print(np.log(sizes2[jj+1]**2) - np.log(sizes2[jj]**2))
            print("__________________________________________")
            print("__________________________________________")


            plt.plot(Data2[jj][0], var1, "-o", label=f"L={sizes2[jj+1]}-{sizes2[jj]}")
            print(Data2[jj][0])
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.semilogx()
        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy\n{dt8[1][0][-1]}D Logarithmic Case - Random Phase Regime")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes1}_randph_{dt8[1][0][-1]}D_Log.png")
        plt.show()
        plt.close()


#       ================================== L Vs. Aveg. EEnt. != MP Plot ==================================
        MP_all = [MP for MP, _ in Data2]
        s_aveg_all = [s_aveg for _, s_aveg in Data2]
        s_aveg_grouped = [list(col) for col in zip(*s_aveg_all)]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))


        for jj in range(len(sizes2)):
            var2 = [(Data2[jj][1][ii])/(np.log(sizes2[jj]))
                    for ii in range(len(Data2[jj][0]))]

            axs[0].plot(Data2[jj][0], var2, "-o", label=f"L={sizes2[jj]}")
            axs[0].set_xlabel("MP")
            axs[0].set_ylabel("E[S]/L")
            axs[0].set_title(f"Measurement Probability vs.\nAveraged Entanglement Entropy {dt8[1][0][-1]}D")

            # Standard Diveation
            var3 = [(Data2[jj][1][ii])/(sizes2[jj]**2)
                    for ii in range(len(Data2[jj][0]))]
            axs[1].plot(Data2[jj][0], var3, "-o", label=f"L={sizes2[jj]}")
            axs[1].set_xlabel("MP")
            axs[1].set_ylabel("E[S]/(L*L)")
            axs[1].set_title(f"Measurement Probability vs.\nAveraged Entanglement Entropy {dt8[1][0][-1]}D")

        fig.suptitle(f'Test')
        plt.savefig(f"plots/tendencies-(1|L, 1|L*L))_L={str(sizes2)}_MP={str(MP)}.png")
        plt.show()
        plt.close()

#       ======================================== Parameter Fitting ========================================
        for jj in range(len(MP)):
            MP_Data2.append([s_aveg_grouped[jj][ii]/sizes2[ii] for ii in range(len(s_aveg_grouped[jj]))])

        A_fits = [[], []] # [ []:A1,
                          #   []:A2 ]

        for jj in range(len(MP)):
#       ================= A1 Fit =================
            A1_aux = MP_Data2[jj][:-1]
            A1_x = np.array(sizes2[:-1], dtype=float)
            
            #print(A1_aux)
            #print(A1_x)
            t = np.log(A1_x)
            # Fit y ≈ A * t + B
            A1, B1 = np.polyfit(t, A1_aux, deg=1)
            #print(A1, B1)

            A_fits[0].append(A1)

#       ================= A2 Fit =================
            A2_aux = MP_Data2[jj][1:]
            A2_x = np.array(sizes2[1:], dtype=float)
            #print(A2_aux)
            #print(A2_x)

            t = np.log(A2_x)
            # Fit y ≈ A * t + B
            A2, B2 = np.polyfit(t, A1_aux, deg=1)
            #print(A2, B2)

            A_fits[1].append(A2)

        plt.plot(MP, A_fits[0],  "-o", label="A1")
        plt.plot(MP, A_fits[1],  "-o", label="A2")

        plt.xlabel("Measurament Probability")
        plt.ylabel("Fitting Parameter Val.")
        plt.legend()
        plt.savefig(f"plots/MPvsFitPar_MP={MP}_Func=A*Log(L)+B_randph_{dt8[1][0][-1]}D.png")
        #plt.show()
        plt.close()


        for jj in range(len(MP)):
            var = [s_aveg_grouped[jj][ii]/sizes2[ii] for ii in range(len(s_aveg_grouped[jj]))]
            plt.plot(sizes2, var,  "-o", label=f"L={MP[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        x = np.linspace(12, 28, 500)
        y = np.log(x)

        plt.plot(x, y, 'k--', label='y = x·log(x)')
        plt.plot(x, x/x, 'g--', label='y = x·log(x)')

        #plt.legend()
        plt.title(f"{dt8[1][0][-1]}D")
        plt.savefig(f"plots/LvsS_Ave_MP={MP}_L={sizes2}_{dt8[1][0][-1]}D.png")
        #plt.show()
        plt.close()



dimbethe= True
sizesbethe = [2,3,4]

Databethe = [] # [(MP:list[float], s_aveg:list[float]):L1, 
           #  (MP:list[float], s_aveg:list[float]):L2, ...]

MP_Databethe = []

if dimbethe:
    print("-2D:")
    for L in tqdm(sizesbethe):

        dt8 = raw_data_reader(L, "bethe", ["tot", "500", "EEqq", "bethe"])

        dt_ave = trajectory_average(dt8[0]) # Mean
        #dt_std = np.std(dt8[0], axis=1) # Standard Deviation

        MP = [kk[0] for kk in dt8[1]] 
        Time = [(kk[1],kk[2]) for kk in dt8[1]] 
        limit = 1
        s_aveg = s_average(dt_ave, limit)
        Databethe.append((MP, s_aveg))
    
        if T_S_plot:
            # Average/Standard Deviation plots
            for ii in range(len(dt_ave)):
                #tt = np.linspace(0, Time[ii][0], Time[ii][0]//Time[ii][1])
                #tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time[ii][0]),0.15)] + \
                #      [Time[ii][0]-1]
                #tt = [500 - ii for ii in range(15)]

                tt = list(range(500))

                # Average
                plt.plot(tt, dt_ave[ii])
                plt.xlabel("time")
                plt.ylabel(r'E[S]')
                plt.title("Mean")

                #fig.suptitle(f'Entanglement Entropy vs. Time {dt8[1][ii][3]}D, Num. of Iters={len(dt8[0][ii])}, MP={format(MP[ii], ".2f")}')
                #plt.savefig(f"plots/EEvsT_L={L}x{L}_Iters={len(dt8[0][ii])}_MP={format(MP[ii], ".2f")}_{dt8[1][ii][3]}D_exp.png")
                plt.show()
                plt.close()
        
        plt.clf()

    if P_S_plot:
        
#       ======================================== P Vs. EEnt. Plot ========================================
        for jj in range(len(sizesbethe)):
            plt.plot(Databethe[jj][0], Databethe[jj][1], "-o", label=f"L={sizesbethe[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Averaged Entanglement Entropy\n{dt8[1][0][-1]}D - Random Phase Regime")
        #plt.savefig(f"plots/MPvsS_Ave_L={sizes2}_randph_{dt8[1][0][-1]}D.png")
        plt.show()
        plt.close()



"""
        # Linear case
        for jj in range(len(sizes2)-1):
            var2 = [(Data2[jj+1][1][ii] - Data2[jj][1][ii])/(sizes2[jj+1] - sizes2[jj])
                   for ii in range(len(Data2[jj][0]))]
            
            plt.plot(Data2[jj][0], var2, "-o", label=f"L={sizes2[jj+1]}-{sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy {dt8[1][0][3]}D Linear Case")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes2}_{dt8[1][0][-1]}D_Linear.png")
        #plt.show()
        plt.close()

        # Quadratic case
        for jj in range(len(sizes2)-1):
            var2 = [(Data2[jj+1][1][ii] - Data2[jj][1][ii])/
                    (sizes2[jj+1]*sizes2[jj+1] - sizes2[jj]*sizes2[jj])
                   for ii in range(len(Data2[jj][0]))]
            
            plt.plot(Data2[jj][0], var2, "-o", label=f"L={sizes2[jj+1]}-{sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy {dt8[1][0][-1]}D Quadratic Case")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes2}_{dt8[1][0][-1]}D_quadr.png")
        plt.close()

        ############################## P Vs. EEnt. Plot ##############################
        for jj in range(len(sizes2)-1):

            var2 = [(Data2[jj+1][1][ii]/sizes2[jj+1] - Data2[jj][1][ii]/sizes2[jj])/(np.log(sizes2[jj+1]) - np.log(sizes2[jj]))
                   for ii in range(len(Data2[jj][0]))]
            
            plt.plot(Data2[jj][0], var2, "-o", label=f"L={sizes2[jj+1]}-{sizes2[jj]}")
            plt.xlabel("Measurament Probability")
            plt.ylabel("E[S]")

        plt.legend()
        plt.title(f"Measurement Probability vs. Variation of Entanglement Entropy {dt8[1][0][-1]}D Quadratic Case")
        plt.savefig(f"plots/MPvsS_Ave_L={sizes2}_{dt8[1][0][-1]}D_quadr.png")
        plt.show()
        plt.close()


"""
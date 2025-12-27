import os
import re
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename:str)->list[list]:
    """
    Takes the values in a file and outputs a list of lists, in which each list has the
    elements of a single line.
    filename: path + name of the file which is to be read
    """
    dt = []
    with open(filename) as f:
        for line in f:
            dt.append(list(map(float, line.split())))
    return dt

def s_average(fl_names:list[str], size:list[int], lmt:float)->list[list]:
    """
    flm_names: List of the names of the elements we want to calculate the entanglement 
               entropy time average of.
    lmt: Index from where we do the time average of the plateu.
    size: List of the sizes of the system in the files.
    """
    # initializing the necessary lists
    s_aveg = [] # output list
    lst_aux1 = [] # takes the whole data from a certain L
    lst_aux2 = [] # takes the list of S averaged in time

    for ii in range(len(fl_names)):
        # taking data of one of the files
        lst_aux1 = read_data(fl_names[ii])
        lst_aux2.append(size[ii])

        # calculation of the value of the time average of the entanglement entropy of 
        # of the platue for different values of measurement probability
        for jj in range(len(lst_aux1)-1):
            ss = np.average(lst_aux1[jj][lmt:])
            lst_aux2.append(ss)

        # inserting the results into the output list with the respective size as the 
        # starting element
        s_aveg.append(lst_aux2.copy())

        # cleaning auxiliary lists
        lst_aux1.clear()
        lst_aux2.clear()
    
    # sorting results by size and removing the size term from the lists
    s_aveg.sort()
    s_aveg = [x[1:] for x in s_aveg]

    return s_aveg

def test_plot(filename:str, log:list[bool]=[False, False])->None:
    """
    filename: Name of the file the data
    log:
    """
    pars = [float(x) if "." in x else int(x) 
            for x in re.findall(r"-?\d+\.?\d*", filename)]
    
    Time = pars[-1]
    
    if "exp" in filename:
        tt = [0,1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + [Time-1]
        name = "_exp"
    else:
        tt = list(range(10000))

    data = read_data(filename)

    for ii in range(len(data)):
        plt.plot(tt, data[ii])
        plt.xlabel("time")
        plt.ylabel("S")
        plt.title(f'Entanglement Entropy vs. Time, Iter = {ii}')

        if log[0]: plt.semilogx()
        if log[1]: plt.semilogy()

        name = "_Iter=" + str(ii) 
        
        if "2D" in filename: name = "plots/test/TvsS_2D_L=" + str(pars[1]) + name
        else: name = "plots/test/TvsS_1D_L=" + str(pars[0]) + name
    
        plt.savefig(name+".png")
        plt.clf()

def S_Time_plotter(filename:str, log=[False, False])->None:
    """
    Plots the whole data of the files as time vs. entanglement entropy plots for each 
    value of measurement probability.
    filename: Name of the file the data
    """
    pars = [float(x) if "." in x else int(x) 
            for x in re.findall(r"-?\d+\.?\d*", filename)]
    
    Time = pars[-1]
    
    if "exp" in filename:
        tt = [0, 1,2,4,6] + [int(np.exp(x)) for x in np.arange(2,np.log(Time),0.15)] + [Time-1]
        name = "_exp"
    else:
        tt = list(range(10000))

    data = read_data(filename)

    for ii in range(len(data)):
        pp = ii*pars[2]
        print(len(data[ii]))
        plt.plot(tt, data[ii])
        plt.xlabel("time")
        plt.ylabel("S")
        plt.title(f'Entanglement Entropy vs. Time, P = {pp}')

        if log[0]: plt.semilogx()
        if log[1]: plt.semilogy()

        name = "_p=" + str(pp) 
        
        if "2D" in filename: name = "plots/TvsS_2D_L=" + str(pars[1]) + name
        else: name = "plots/TvsS_1D_L=" + str(pars[0]) + name
    
        plt.savefig(name+".png")
        plt.clf()

def P_S_plotter(s_aveg:list[list], L:list[int], pmax:float, pstep:float, dim2:bool=False, show=False)->None:
    """

    """
    for kk in range(len(s_aveg)):
        plt.plot(np.arange(0, pmax, pstep), [s_aveg[kk][ii]/np.log(L[kk]) for ii in range(len(s_aveg[kk]))], "o-", label = "L = " + 
                 str(L[kk]))

    plt.ylabel("S")
    plt.xlabel("Measurement Prob.")
    plt.legend()
    name = "plots/probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) + ")"
    if dim2: name += "_2D"
    if show: plt.show()
    plt.savefig(name+".png")
    plt.clf()

def calc_log(ii, jj, l1, l2):
    num = ii-jj
    de = np.log(l1)-np.log(l2)
    return num/de

def calc(ii, jj, l1, l2):
    num = ii-jj
    de = l1-l2
    return num/de

####################################### Systems ########################################

dim_1 = False
dim_2 = True

fl_names = os.listdir("data/")

fl_names2 = ["data/"+x for x in fl_names if "EEqq" and "2D" in x]

lst1 = []
lst2 = []
lst3 = []

pars = [
        [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+\.?\d*", s)]
        for s in fl_names2
    ]

L = [x[1] for x in pars]

print(L)

lst1 = s_average(fl_names2, L,  1000)

for ii in range(len(lst1)):
    lst1[ii].insert(L[ii], 0)

lst1.sort()
L.sort()

lst1 = [x[1:] for x in lst1]


for ii in range(len(lst1)-1):
    print(L[ii+1], L[ii])
    for jj in range(len(lst1[0])):
        lst2.append(calc_log( lst1[ii+1][jj], lst1[ii][jj], L[ii+1], L[ii]))
    
    lst3.append(lst2.copy())
    lst2.clear()


for ii in lst3:
    plt.plot([0.05*ii for ii in range(0,8)], ii, "-o")

plt.show()
plt.clf()

if dim_1:

    fl_names2 = ["data/"+x for x in fl_names if "EEqq" and not "2D" in x]

    
    for ii in fl_names2:
        S_Time_plotter(ii, [False, False])

    pars = [
        [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+\.?\d*", s)]
        for s in fl_names2
    ]

    if all(x[1:] == pars[0][1:] for x in pars):
        L = [x[0] for x in pars]

        # parameters
        pmax = pars[0][1]
        pstep = pars[0][2]
        lmt = 1000

        # entanglement entropy time average calculation
        s_aveg = s_average(fl_names2, L, lmt)
        L.sort()

        # plotting
        P_S_plotter(s_aveg, L, pmax, pstep)
        
if dim_2:

    fl_names = ["data/"+x for x in fl_names if "EEqq" and "2D" in x]

    for ii in fl_names:
        S_Time_plotter(ii, [False, False])


    pars = [
        [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+\.?\d*", s)]
        for s in fl_names
    ]

    if all(x[3:] == pars[0][3:] for x in pars):
        L = [x[1]*x[2] for x in pars]

        # parameters
        pmax = pars[0][3]
        pstep = pars[0][4]
        print(pmax, pstep)
        lmt = 1000

        # entanglement entropy time average calculation
        s_aveg = s_average(fl_names, L, lmt)
        L.sort()
        
        P_S_plotter(s_aveg, L, pmax, pstep, True, show=True)
        
        """
        # plotting
        for kk in range(len(s_aveg)):
            plt.plot(np.arange(0, pmax, pstep), s_aveg[kk], "o-", label = "L = " + 
                    str(L[kk]))
        plt.legend()
        plt.savefig("probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) 
                    + ")_2D.png")
        """

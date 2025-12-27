import numpy as np
import re
import os
from typing import Callable
from collections import defaultdict
import matplotlib.pyplot as plt

###################################### Reader ##########################################

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

def raw_data_reader(size:int, dim:str, kw:list[str]=[], Nkw:list[str]=[], lmbd:Callable[[float], float]=None)->tuple[list]:
    """
    Gathers all files with the raw data from systems with size "size" and with different 
    values of measurement probability (mp) and put them all in a list. Gives as output a
    tuple in which the 1st elments is the data divided by different values of mp and the
    2nd the parameters of each element in data. 
    - size: int which represents the size of the system for which we want to extract the
            raw that from
    """
    # File names in data directory
    if dim == "1D": dir_name = f"data/1D/syst_L={size}"
    if dim == "2D": dir_name = f"data/2D/syst_L={size}x{size}"
    if dim == "bethe": dir_name = f"data/bethe/syst_bethe_n={size}"

    fl_names = os.listdir(dir_name+"/")

    if kw: fl_names = [t for t in fl_names if all(k in t for k in kw)]
    if Nkw: fl_names = [t for t in fl_names if all(not k in t for k in Nkw)]

    print(fl_names)
    data = [] # raw data organized such that: 
              # [[iter1,iter2,...]:Meas.Prob.1, [[iter1,iter2,...]]::Meas.Prob.2,...]

    #       parameter extraction and verification:
    pars = [
        [
            float(f"{float(x):.2f}") if "." in x else int(x)
            for x in re.findall(r"-?\d+\.?\d*", s)
        ]
        for s in fl_names
    ]
    
    fls = [dir_name+"/"+ x for x in fl_names] 
    pars = [x[1:] for x in pars] # takes the system size parameter

    #       file reading
    lst_aux1 = []
    lst_aux2 = []

    for ii in range(len(fls)):
        lst = read_data(fls[ii])

        # Direct alteration of the values calculated
        if lmbd:
            for jj in range(len(lst)):
                lst_aux2 = [lmbd(xx) for xx in lst[jj]]
                lst_aux1.append(lst_aux2.copy())
                lst_aux2.clear()

            lst = lst_aux1.copy()
            lst_aux1.clear()

        lst.insert(0,pars[ii][0])
        data.append(lst)

    #       sort by probability measurement value
    data.sort()
    data = [x[1:] for x in data] # takes the measurement probability parameter
    pars.sort()

    return data, pars

def trajectory_average(data:list)->list:
    ave_trajs = []

    for pp in range(len(data)):
        ave = np.mean(data[pp], axis=0)
        ave_trajs.append(list(ave))

    return ave_trajs    
    
def s_average(data:list[str], lmt:float)->list[list]:
    """
    lmt: Index from where we do the time average of the plateu.
    """
    # initializing the necessary lists
    s_aveg = [] # output list

    for ii in range(len(data)):
        s_aveg.append(np.average(data[ii][lmt:]))     

    return s_aveg

def invert_data_structure(data_list, size_list):
    grouped_data = defaultdict(lambda: ([], []))

    for (mp_list, s_aveg_list), l_value in zip(data_list, size_list):
        for mp, s in zip(mp_list, s_aveg_list):
            storage = grouped_data[mp]
            storage[0].append(l_value)
            storage[1].append(s)

    data_pairs = []
    mp_keys = []
    
    sorted_mps = sorted(grouped_data.keys())

    for mp in sorted_mps:
        if mp == 0: continue
        if mp * 100 % 5 == 0 and mp < 0.5:
            l_values, s_values = grouped_data[mp]
            data_pairs.append((l_values, s_values))
            mp_keys.append(mp)

    return data_pairs, mp_keys

def plot_mp_vs_var_quant(Data1:list[tuple], sizes1:list[int],  
                         plot_title:str,    filename:str,   
                         ylabel:str="",     xlimits:tuple=(), 
                         xlabel:str="Measurament Probability", func:Callable[[float], float]=None):

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for jj in range(len(sizes1)-1):
        x_prev, y_prev = Data1[jj]
        x_curr, y_curr = Data1[jj+1]

        common_x = sorted(list(set(x_prev).intersection(set(x_curr))))

        map_prev = {x: y for x, y in zip(x_prev, y_prev)}
        map_curr = {x: y for x, y in zip(x_curr, y_curr)}

        y_prev_filtered = [map_prev[x] for x in common_x]
        y_curr_filtered = [map_curr[x] for x in common_x]

        dem = sizes1[jj+1] - sizes1[jj]
        if func: denom = func(sizes1[jj+1]) - func(sizes1[jj])
        
        var1 = [(y_curr_filtered[i] - y_prev_filtered[i]) / denom for i in range(len(common_x))]
        
        plt.plot(common_x, var1, "-o", label=f"L={sizes1[jj+1]} - {sizes1[jj]}")
    
    if xlimits: plt.xlim(xlimits[0], xlimits[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(plot_title)
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_mp_vs_quant(Data1:list[tuple], sizes1:list[int],  
                     plot_title:str,    filename:str,   
                     ylabel:str="",     xlimits:tuple=(), 
                     xlabel:str="Measurament Probability", func:Callable[[float], float]=None):
    
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    for jj in range(len(sizes1)):
        var1 = Data1[jj][1]
        if func: var1 = [func(Data1[jj][1][ii]) for ii in range(len(Data1[jj][0]))]
        plt.plot(Data1[jj][0], Data1[jj][1], "-o", label=f"L={sizes1[jj]}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    if xlimits: plt.xlim(xlimits[0], xlimits[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(plot_title)
    plt.savefig(filename)
    plt.show()
    plt.close()


""""
calc log and without log: X

Repeat the calculation and average for 30 X

Write the code for 2D square lattice and periodic boundaries 

Make something like a neel state one line and the other shifted one

Number the lattice like a 1D string but dont forget the links. Diag matrix between lines and do the layers: X
"""



import os
import re
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
    dt = []
    with open(filename) as f:
        for line in f:
            dt.append(list(map(float, line.split())))
    return dt

def calc_log(ii, jj, l1, l2):
    num = ii-jj
    de = np.log(l1)-np.log(l2)
    return num/de

def calc(ii, jj, l1, l2):
    num = ii-jj
    de = l1-l2
    return num/de


fl_names = os.listdir("data/")
fl_names = [x for x in fl_names if "EEqq" in x and not "2D" in x and "mean" in x]
print(fl_names)

lmt = 1000
pmax = 0.4
pstep = 0.05
L = [8, 16, 32, 64]
p = np.arange(0, pmax, pstep)
print(p)

data = []
s_aveg = []

lst_aux1 = [] # takes the whole data from a certain L
lst_aux2 = [] # takes the list of S averaged in time

for ii in fl_names:
    lst_aux1 = read_data("data/"+ii)

    size = [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+\.?\d*", ii)]
    lst_aux2.append(size[0])
    print(size)

    for jj in range(len(lst_aux1)-1):

        print(jj+1)

        ss = np.average(lst_aux1[jj][lmt:])

        lst_aux2.append(ss)

    s_aveg.append(lst_aux2.copy())
 
    lst_aux1.clear()
    lst_aux2.clear()

s_aveg.sort()
s_aveg = [list(x[1:]) for x in s_aveg]

qt_log = []
qt = []
lst_aux3 = []
lst_aux4 = []

for ll in range(len(L)-1):
    lst_aux3 = [calc_log(s_aveg[ll][x], s_aveg[ll+1][x], L[ll], L[ll+1]) for x in range(len(p))]
    lst_aux4 = [calc(s_aveg[ll][x], s_aveg[ll+1][x], L[ll], L[ll+1]) for x in range(len(p))]
    qt_log.append(lst_aux3.copy())
    qt.append(lst_aux4.copy())
    lst_aux3.clear()
    lst_aux4.clear()

print("_________________________________________________________")

print(len(s_aveg), len(s_aveg[0]))
for kk in range(len(s_aveg)):

    plt.plot(p, s_aveg[kk], "o-", label = "L = " + str(L[kk]))

plt.legend()
plt.show()
plt.savefig("probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) +").png")

plt.clf()

for kk in range(len(L)-1):

    plt.plot(p, qt_log[kk], "o-", label = "L = " + str(L[kk]) + " - " + str(L[kk+1]) )

plt.title("Log")
plt.legend()
plt.show()
plt.savefig("probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) +")_qt_log.png")

plt.clf()

for kk in range(len(L)-1):

    plt.plot(p, qt[kk], "o-", label = "L = " + str(L[kk]) + " - " + str(L[kk+1]) )

plt.title("Size")
plt.legend()
plt.show()
plt.savefig("probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) +")_qt.png")



  
"""    
p = np.arange(0, pmax, pstep)

    data = []
    s_aveg = []

    lst_aux1 = [] # takes the whole data from a certain L
    lst_aux2 = [] # takes the list of S averaged in time

    for ii in fl_names:
        lst_aux1 = read_data(ii)
        size = [float(x) if "." in x else int(x) for x in re.findall(r"-?\d+\.?\d*", ii)]
        lst_aux2.append(size[0])

        for jj in range(len(lst_aux1)-1):
            ss = np.average(lst_aux1[jj][lmt:])
            lst_aux2.append(ss)

        s_aveg.append(lst_aux2.copy())
        lst_aux1.clear()
        lst_aux2.clear()

    s_aveg.sort()
    s_aveg = [list(x[1:]) for x in s_aveg]

print("_________________________________________________________")

for kk in range(len(s_aveg)):

    plt.plot(p, s_aveg[kk], "o-", label = "L = " + str(L[kk]))

plt.legend()
plt.savefig("plots/probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) +").png")
"""



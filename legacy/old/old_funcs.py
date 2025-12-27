def test_plot(filename:str, log:list[bool]=[False, False])->None:
    """
    filename: Name of the file the data
    log: list which controls if one wants the x axis (log[0]) and/or the y axis (log[1])
         in logarithmic scale.
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
    print(data[0])

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
    log: list which controls if one wants the x axis (log[0]) and/or the y axis (log[1])
         in logarithmic scale.
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
    print(data[0])

    for ii in range(len(data)):
        pp = ii*pars[-2]

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


def P_S_plotter(s_aveg:list[list], L:list[int], pmax:float, pstep:float)->None:
    """
    - s_aveg: list of the averaged plataue of the entanglement entropy per measurement 
              probability value per system size.
    - L: list of the sizes of the systems presented in the s_aveg list.
    - pmax: maximum value of the measurement probability.
    - pstep: step of the measurement probability.
    """
    for kk in range(len(s_aveg)):
        plt.plot(np.arange(0, pmax, pstep), s_aveg[kk], "o-", label = "L = " + 
                 str(L[kk]))

    plt.ylabel("S")
    plt.xlabel("Measurement Prob.")
    plt.legend()
    plt.savefig("plots/probVS.s_L=" + str(L) + "_p=(" +str(pmax) + "," + str(pstep) 
            + ").png")
    plt.clf()


def P_S_plotter_func(s_aveg:list[list], L:list[int], pmax:float, pstep:float, 
                     funcs:list, func_names:list[str] = [])->None:
    """
    - s_aveg: list of the averaged plataue of the entanglement entropy per measurement 
              probability value per system size.
    - L: list of the sizes of the systems presented in the s_aveg list.
    - pmax: maximum value of the measurement probability.
    - pstep: step of the measurement probability.
    - funcs: list of lambda functions. These lambda functions are organized in the
             following way:
             ff(list[list]1, list[list]2, size1, size2)
    """
    for ff in range(len(funcs)):

        for kk in range(len(s_aveg)-1):
            plt.plot(np.arange(0, pmax, pstep), funcs [ff](s_aveg[kk], s_aveg[kk+1], 
                                                           L[kk], L[kk+1]), "o-", 
                                                           label = f'L ={str(L[kk])},\
                                                           {str(L[kk+1])}')

        if func_names: plt.ylabel(func_names[ff])  
        else: plt.ylabel("S")  

        plt.xlabel("Measurement Prob.")
        plt.legend()
        plt.savefig(f'plots/probVS.s_L={str(L)}_p=({str(pmax)},{str(pstep)})_func={ff}\
                    .png')
        plt.clf()


def flatten_matrix(M):
    return np.reshape(M, -1)

def unflatten_vector(v, shape):
    return np.reshape(v, shape)

def dCdt(t, c_flat):
    C = unflatten_vector(c_flat, H.shape) # assuming H and C are square matrices of the same shape
    dC = - np.dot(C,H) - np.dot(H,C) + 2*np.dot(C, np.dot(H,C))
    return flatten_matrix(dC)

def brick_structure_pair_evol(N:int, pair:bool)->np.ndarray:
    """

    N:
    pair:
    """
    A = np.zeros((N,N), dtype=complex)
    
    if pair:
        for jj in range(0, N, 2):
            A[jj:jj+2, jj:jj+2] = unitary_group.rvs(2)
    else:      
        for kk in range(1, N-1, 2):
            A[kk:kk+2, kk:kk+2] = unitary_group.rvs(2)
        A[np.ix_([0,N-1], [0,N-1])] = unitary_group.rvs(2)

    return A

def brick_structure_odd_evol(N:int, iter_N:int)->np.ndarray:
    """
    Creates 
    N:
    pair:
    """
    A = np.identity(N, dtype=complex)
    
    if iter_N==1:
        for ii in range(0, N-1, 2):
            A[ii:ii+2,ii:ii+2] += unitary_group.rvs(2)
            print(ii)

    elif iter_N%2==0:
        pos = iter_N - 2
        for ii in range(0, N-1, 2):
            pivot = ii + 1 + pos
            print(pivot)
            if pivot >= N: pivot = 0
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

    if iter_N%2==1 and iter_N!=1:
        N_blocks_odd = (N-1)//2 - 1
        pos = iter_N -2

        A[np.ix_([0,N-1], [0,N-1])] = unitary_group.rvs(2)

        for jj in range(N_blocks_odd):
            pivot = pos + 1 + jj*2
            if pivot >= N-1: pivot = N - pivot
            A[pivot:pivot+2,pivot:pivot+2] += unitary_group.rvs(2)

    return A

def entanglement_entropy_free(h:np.ndarray, subsystem:list, occ_modes:list)->tuple:

    L = h.shape[0]
    e, v = np.linalg.eigh(h)  # e: eigenvalues, v: eigenvectors (columns)

    # Build correlation matrix for the occupied modes
    v_occ = v[:, occ_modes]
    print(v_occ)
    C = v_occ @ v_occ.conj().T
    print("Correlation matrix C:\n", C)

    # Restrict to subsystem A
    CA = C[np.ix_(subsystem, subsystem)]
    z, _ = np.linalg.eigh(CA)

    # Clip eigenvalues to avoid numerical issues
    eps = 1e-12
    z = np.clip(z, eps, 1-eps)

    # Von Neumann entanglement entropy
    S = -np.sum(z * np.log(z) + (1-z) * np.log(1-z))
    return S, z
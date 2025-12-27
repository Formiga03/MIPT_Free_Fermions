"""
N = 10
- Eigenvals comp. 10x10: 0.0005 s
N = 50
- Eigenvals comp. 50x50: 0.0081 s
N = 100
- Eigenvals comp. 100x100: 0.9238 s
N = 500
- Eigenvals comp. 500x500: 1.4931 s
N = 1000
- Eigenvals comp. 1000x1000: 6.9853 s
N = 1500
- Eigenvals comp. 1500x1500: 18.1995 s
N = 2000
- Eigenvals comp. 2000x2000: 45.8365 s
N = 2500
- Eigenvals comp. 2500x2500: 85.8795 s
N = 3000
- Eigenvals comp. 3000x3000: 153.8271 s
N = 3500
- Eigenvals comp. 3500x3500: 233.8620 s
N = 4000
- Eigenvals comp. 4000x4000: 328.6584 s
"""

import os
import numpy as np, time
import matplotlib.pyplot as plt
import mkl  # requires mkl-service

# Optional: control threading to make runs comparable/reproducible
# You can change this to e.g. 16, 8, 4, 1
TARGET_THREADS = 16
mkl.set_num_threads(TARGET_THREADS)

print("MKL get_max_threads():", mkl.get_max_threads())

# For clarity also enforce via env (must happen before heavy compute ideally)
os.environ["MKL_NUM_THREADS"] = str(TARGET_THREADS)
os.environ["OMP_NUM_THREADS"] = str(TARGET_THREADS)

# Show how NumPy was built (should mention mkl-sdl)
np.__config__.show()

data = []
sizes = [10, 50, 100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]

for n in sizes:
    rng = np.random.default_rng(12345)

    # Generate one complex matrix of size n x n
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))

    print(f"N = {n}")
    t0 = time.perf_counter()
    for ii in range(10):
        vct = np.linalg.eigvals(A)
    t1 = time.perf_counter()

    elapsed = t1 - t0
    print(f"- Eigenvals comp. {n}x{n}: {elapsed:.4f} s")

    # store average per eigvals() call
    data.append(elapsed / 10.0)

# Plot timing vs matrix size
plt.figure()
plt.plot(sizes, data, "-o")
plt.xlabel("Matrix Size")
plt.ylabel("Average eigvals() Time (s)")
plt.grid()
plt.tight_layout()
plt.savefig("linalg.eigvals_efficiency_per_mat_size_mkl.png")

"""
n = 1000  # you can try 4096 if you have enough RAM
rng = np.random.default_rng(12345)
# warmup
A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
B = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))

t0 = time.perf_counter()
for ii in range(1000):
    print(ii)
    C = np.dot(A,B)
t1 = time.perf_counter()
print(f"MatMul {n}x{n}: {(t1 - t0)/1000:.4f} s")
print(C)


###################################################################################################################
n = 10000  # you can try 4096 if you have enough RAM
rng = np.random.default_rng(0)

t0 = time.perf_counter()

for ii in [100000000]:
    # warmup
    A = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    B = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    C = A @ B

t1 = time.perf_counter()

print(f"MatMul {n}x{n}: {t1 - t0:.4f} s")

H = 0.5*(A + A.conj().T)  # make Hermitian
_ = np.linalg.eigh(H)

t2 = time.perf_counter()
w, V = np.linalg.eigh(H)
t3 = time.perf_counter()

print(f"Hermitian eig {n}x{n}: {t3 - t2:.4f} s")
print("First 3 eigenvalues:", w[:3].real)
mport numpy as np

lst = [[[1,2,3,4],[1,3,4,5],[2,1,4,5]],
       [[2,1,3,5],[2,3,4,5],[2,1,5,3]]]

mean = np.mean(lst, axis=1)
print(mean)

sd1 = np.std(lst, axis=1)
sd2 = np.std(lst, axis=2)

print(sd1)
print(sd2)

import networkx as nx

G = nx.Graph()
G.add_edge("A", "B", weight=4)
G.add_edge("B", "D", weight=2)
G.add_edge("A", "C", weight=3)
G.add_edge("C", "D", weight=4)
print(G.graph)
print(nx.shortest_path(G, "A", "D", weight="weight"))

import os

fl_names = os.listdir("data/")

fls = [x for x in fl_names if "2D" not in x]

for ii in fls:
    old_name = "data/" + ii
    new_name = old_name + "_1D"

    os.rename(old_name, new_name)
    

"""

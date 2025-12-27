import numpy as np, time
import matplotlib.pyplot as plt

# Show how NumPy was built (should NOT mention mkl; likely "blas: system" or OpenBLAS)
np.__config__.show()

# If mkl-service is not installed here, this import would fail.
# We intentionally do NOT import or configure mkl in this script.

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
plt.savefig("linalg.eigvals_efficiency_per_mat_size_no_mkl.png")

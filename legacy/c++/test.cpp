#include <mkl.h>
#include <vector>
#include <iostream>
#include <random>
#include <iomanip>
#include <chrono>
using namespace std;

// Helper to build complex numbers
inline MKL_Complex16 cz(double re, double im) {
    MKL_Complex16 z;
    z.real = re;
    z.imag = im;
    return z;
}

// Pretty-print complex numbers
void print_c(const MKL_Complex16 &z) {
    std::cout << "("
              << z.real
              << (z.imag >= 0 ? "+" : "")
              << z.imag << "i)";
}

// Compute eigenvalues and right eigenvectors of a general complex matrix
// A_in is n x n, row-major
// eigVals[j] = eigenvalue λ_j
// eigVecs(:,j) = right eigenvector for λ_j
MKL_INT diagonalise_general(
    MKL_INT n,
    const std::vector<MKL_Complex16> &A_in,
    std::vector<MKL_Complex16> &eigVals,
    std::vector<MKL_Complex16> &eigVecs
) {
    // LAPACKE_zgeev overwrites A, so make a copy
    std::vector<MKL_Complex16> A = A_in;

    eigVals.resize(n);
    eigVecs.resize(n * n);

    MKL_Complex16 *vl   = nullptr; // we don't want left eigenvectors
    MKL_INT        ldvl = 1;       // must still be >=1 even if vl == nullptr

    MKL_Complex16 *vr   = eigVecs.data(); // right eigenvectors
    MKL_INT        ldvr = n;              // leading dimension for vr in row-major

    MKL_INT info = LAPACKE_zgeev(
        LAPACK_ROW_MAJOR, // row-major layout
        'N',              // jobvl: no left eigenvectors
        'V',              // jobvr: yes right eigenvectors
        n,
        A.data(),         // input matrix (will be destroyed)
        n,                // lda
        eigVals.data(),   // eigenvalues λ_j (complex)
        vl,               // left eigenvectors (unused)
        ldvl,             // must be >=1
        vr,               // right eigenvectors
        ldvr              // ldvr
    );

    return info;
}

int main() {
    // Choose matrix size
    const MKL_INT n = 5000;

    // Build a random complex n x n matrix in row-major
    std::vector<MKL_Complex16> A(n * n);

    std::mt19937 rng(12345); // deterministic seed
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (MKL_INT i = 0; i < n; ++i) {
        for (MKL_INT j = 0; j < n; ++j) {
            double re = dist(rng);
            double im = dist(rng);
            A[i*n + j] = cz(re, im); // row-major: [i*n + j]
        }
    }

    std::vector<MKL_Complex16> eigVals;
    std::vector<MKL_Complex16> eigVecs;

    auto t0 = std::chrono::high_resolution_clock::now();
    MKL_INT info = diagonalise_general(n, A, eigVals, eigVecs);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = t1 - t0;
    cout << "Time: " << elapsed.count()  << endl;

    if (info != 0) {
        std::cerr << "Eigen solve failed, info = " << info << "\n";
        return 1;
    }


/*
    // Optional: quick residual check for eigenpair j=0:
    // Compute A * v0 - λ0 * v0 and print its norm.
    {
        MKL_INT j = 0;
        std::vector<MKL_Complex16> v0(n);
        for (MKL_INT i = 0; i < n; ++i)
            v0[i] = eigVecs[i*n + j];

        // r = A*v0 - λ0*v0
        std::vector<MKL_Complex16> r(n, cz(0.0,0.0));
        for (MKL_INT i = 0; i < n; ++i) {
            // compute (A*v0)[i]
            double rr = 0.0, ri = 0.0;
            for (MKL_INT k = 0; k < n; ++k) {
                // complex multiply A[i,k] * v0[k]
                double ar = A[i*n + k].real;
                double ai = A[i*n + k].imag;
                double br = v0[k].real;
                double bi = v0[k].imag;
                rr += ar*br - ai*bi;
                ri += ar*bi + ai*br;
            }
            // subtract λ0 * v0[i]
            double lr = eigVals[j].real;
            double li = eigVals[j].imag;
            double vr = v0[i].real;
            double vi = v0[i].imag;
            rr -= (lr*vr - li*vi);
            ri -= (lr*vi + li*vr);

            r[i] = cz(rr, ri);
        }

        // compute ||r||_2
        double norm2 = 0.0;
        for (MKL_INT i = 0; i < n; ++i) {
            double rr = r[i].real;
            double ri = r[i].imag;
            norm2 += rr*rr + ri*ri;
        }
        std::cout << "\nResidual norm for eigenpair j=0: "
                  << std::sqrt(norm2) << "\n";
    }
*/
    return 0;
}

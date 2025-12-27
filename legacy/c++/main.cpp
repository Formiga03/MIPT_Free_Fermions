#include <mkl.h>     // mkl.h location/activation code: source /opt/intel/oneapi/setvars.sh
                     // export OMP_NUM_THREADS=16
                     // export MKL_NUM_THREADS=16
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>
#include <chrono>    
#include <cstdlib>   // std::atoi
#include <cmath>
using namespace std;

// Helper to build MKL_Complex16
inline MKL_Complex16 cz(double re, double im) {
    MKL_Complex16 z;
    z.real = re;
    z.imag = im;
    return z;
}

// Pretty print a matrix (row-major)
void print_mat(const char* name,
               const MKL_Complex16* mat,
               MKL_INT rows,
               MKL_INT cols)
{
    std::cout << name << " =\n";
    for (MKL_INT i = 0; i < rows; ++i) {
        for (MKL_INT j = 0; j < cols; ++j) {
            const MKL_Complex16 v = mat[i * cols + j];
            std::cout << std::fixed << std::setprecision(4)
                      << "(" << v.real
                      << (v.imag >= 0 ? "+" : "")
                      << v.imag << "i)  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
/*
void SQRT_COMP_Mat_Mult(vector<MKL_Complex16> A, vector<MKL_Complex16> B, vector<MKL_Complex16> C)
{

    MKL_INT N = A.size();

    cblas_zgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    M,          // rows of A / C
    N,          // cols of B / C
    K,          // cols of A / rows of B
    &alpha,
    A.data(), lda,
    B.data(), ldb,
    &beta,
    C.data(), ldc
    );
}
*/

int main(int argc, char** argv) {
    // Read matrix dimension N from argv (square matrices N x N)
    // Default N = 2 if not provided
    MKL_INT N = 1000;

    const MKL_INT M = N; // rows of A and C
    const MKL_INT K = N; // cols of A, rows of B
    // cols of B and C is also N

    std::cout << "Generating random " << N << "x" << N
              << " complex matrices A and B, and computing C = A * B\n\n";

    // RNG setup (deterministic seed for reproducibility)
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Allocate A, B, C on the heap (via std::vector)
    std::vector<MKL_Complex16> A(M * K);
    std::vector<MKL_Complex16> B(K * N);
    std::vector<MKL_Complex16> C(M * N);

    // Fill A and B with random values; init C to 0
    for (MKL_INT i = 0; i < M; ++i) {
        for (MKL_INT j = 0; j < K; ++j) {
            double re = dist(rng);
            double im = dist(rng);
            A[i * K + j] = cz(re, im);
        }
    }

    for (MKL_INT i = 0; i < K; ++i) {
        for (MKL_INT j = 0; j < N; ++j) {
            double re = dist(rng);
            double im = dist(rng);
            B[i * N + j] = cz(re, im);
        }
    }

    for (MKL_INT idx = 0; idx < M * N; ++idx) {
        C[idx] = cz(0.0, 0.0);
    }

    cout << sqrt(A.size()) << endl;

 
    // alpha and beta
    MKL_Complex16 alpha = cz(1.0, 0.0);
    MKL_Complex16 beta  = cz(0.0, 0.0);

    // Leading dimensions for row-major (no transpose):
    // lda = K, ldb = N, ldc = N
    const MKL_INT lda = K;
    const MKL_INT ldb = N;
    const MKL_INT ldc = N;
    // warm-up (not timed)
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        M, N, K,
        &alpha,
        A.data(), lda,
        B.data(), ldb,
        &beta,
        C.data(), ldc
    );

    // timed loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int ii=0; ii<1000; ++ii) {
        cblas_zgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K,
            &alpha,
            A.data(), lda,
            B.data(), ldb,
            &beta,
            C.data(), ldc
        );
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double avg = std::chrono::duration<double>(t1 - t0).count() / 1000.0;
    std::cout << "Avg per GEMM: " << avg << " s\n";

    // Compute elapsed seconds as double
    std::chrono::duration<double> elapsed = t1 - t0;
    cout << "Time: " << elapsed.count()/1000  << endl;

    // Print results
    // Be aware: for large N this will spam output. You can comment these out if needed.
    //print_mat("A", A.data(), M, K);
    //print_mat("B", B.data(), K, N);
    //print_mat("C = A * B", C.data(), M, N);

    return 0;
}

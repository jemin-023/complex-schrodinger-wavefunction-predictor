#include <vector>
#include "cnpy.h"
#include <cmath>

// ===== Global Variables =====
constexpr int N = 256;
constexpr double xmin = -5.0;
constexpr double xmax = 5.0;

const double dx = (xmax - xmin)/(N - 1);

//  ====================================================================================================================
//  !! Sturm count is a algorithm used to get the number of eigen values under mu without actually finding the values.!!
//  ====================================================================================================================

// diag: diagonal of H (size N)
// e:    off-diagonal value (scalar, same everywhere)
// mu:   test value
// returns: number of eigenvalues < mu

int sturm_count(const std::vector<double>& diag, double e, double mu){
    int count = 0;
    double s = diag[0] - mu;
    if(s < 0) count++;

    for(int i = 1; i < N; i++){
        if(s == 0.0) s = 1e-14;         // prevent division by zero
        s = (diag[i] - mu) - (e*e)/s;   // recurSON!
        if(s < 0) count++;
    }
    return count;
}


//  ===============================================================
//  !! Bisection is essentially binary search on the energy axis.!!
//  ===============================================================

// diag: diagonal of H = T_diag + V  (size N)
// e:    off-diagonal (scalar = -1/dx²)
// n:    no. of eigen values we want(1, 2, 3... 256) [ in this project n = 18 for no apparent reason ]
constexpr int K = 18;
// returns: Eₙ to high precision


double bisect(const std::vector<double>& diag, double e, double low, double high, int n){
    while(high - low > 1e-10){
        double mid = low + (high - low) / 2.0;
        if(sturm_count(diag, e, mid) > n) high = mid;
        else low = mid;
    }
    return (low + high) / 2.0;
}

// this will give a array of 18 eigen values
std::vector<double> find_eigenvalues(const std::vector<double>& diag, double e){

    // Gershgorin Circle Theorem.

    double low = diag[0] - std::abs(e);
    double high = diag[0] + std::abs(e);
    for(int i = 1; i < N; i++){
        double r = (i == N-1) ? std::abs(e) : 2.0 * std::abs(e);
        low  = std::min(low,  diag[i] - r);
        high = std::max(high, diag[i] + r);
    }

    std::vector<double> eigenvalues(K);
    for(int n = 0; n < K; n++){
        eigenvalues[n] = bisect(diag, e, low, high, n);
    }
    return eigenvalues;
}

//  ========================================================
//  !!Thomas Algorithm Solves Ax=b where, A is tridiagonal!!
//  ========================================================

// solves (H - shift*I)x = b
// diag: diagonal of H
// e: off-diagonal scalar
// b: right hand side vector
// shift: the eigenvalue Eₙ

std::vector<double> thomas(const std::vector<double>& diag, double e, double shift, const std::vector<double>& b){
    std::vector<double> d(N), rhs(N), x(N);

    for(int i = 0; i < N; i++){
        d[i] = diag[i] - shift;
        rhs[i] = b[i];
    }

    // forward elimination
    for(int i = 1; i < N; i++){
        double w = e / d[i-1];
        d[i]   -= w * e;
        rhs[i] -= w * rhs[i-1];
    }

    // back substitution
    x[N-1] = rhs[N-1] / d[N-1];
    for(int i = N-2; i >= 0; i--){
        x[i] = (rhs[i] - e * x[i+1]) / d[i];
    }
    return x;
}


//  ==================================================================================
//  !! Inverse iteration: converges to eigenvector for a known (approximate) eigenvalue
//  ==================================================================================

// diag:  diagonal of H (size N)
// e:     off-diagonal scalar
// eigenvalue: the approximate eigenvalue Eₙ
// returns: normalised eigenvector (size N)

constexpr int MAX_ITER = 100;
constexpr double TOL = 1e-12;

std::vector<double> inverse_iteration(const std::vector<double>& diag, double e, double eigenvalue){
    // start with a random-ish initial vector (all ones, then normalise)
    std::vector<double> x(N, 1.0);
    double norm = 0.0;
    for(int i = 0; i < N; i++) norm += x[i] * x[i];
    norm = std::sqrt(norm);
    for(int i = 0; i < N; i++) x[i] /= norm;

    for(int iter = 0; iter < MAX_ITER; iter++){
        // solve (H - eigenvalue*I) * y = x
        std::vector<double> y = thomas(diag, e, eigenvalue, x);

        // normalise y
        norm = 0.0;
        for(int i = 0; i < N; i++) norm += y[i] * y[i];
        norm = std::sqrt(norm);
        for(int i = 0; i < N; i++) y[i] /= norm;

        // check convergence: |y - x| < TOL  (or |y + x| for sign flip)
        double diff_pos = 0.0, diff_neg = 0.0;
        for(int i = 0; i < N; i++){
            diff_pos += (y[i] - x[i]) * (y[i] - x[i]);
            diff_neg += (y[i] + x[i]) * (y[i] + x[i]);
        }
        x = y;
        if(std::min(diff_pos, diff_neg) < TOL) break;
    }
    return x;
}


//  ===========
//  !! Main() !!
//  ===========

constexpr int M = 10000;  // number of potentials

int main(){
    // 1. load T  (256×256 flat row-major)
    cnpy::NpyArray T_arr = cnpy::npy_load("../data/T.npy");
    double* T_data = T_arr.data<double>();

    double e = T_data[0 * N + 1];           // off-diagonal scalar
    std::vector<double> T_diag(N);
    for(int i = 0; i < N; i++) T_diag[i] = T_data[i * N + i];

    // 2. load potentials  (10000×256)
    cnpy::NpyArray V_arr = cnpy::npy_load("../data/potentials.npy");
    double* V_data = V_arr.data<double>();

    // 3. output storage
    std::vector<double> all_eigenvalues;
    std::vector<double> all_eigenvectors;
    all_eigenvalues.reserve(M * K);
    all_eigenvectors.reserve(M * K * N);

    // 4. loop over M potentials
    for(int m = 0; m < M; m++){
        // a. extract V for this potential → pointer to row m
        double* V = V_data + m * N;

        // b. build diag = T_diag + V
        std::vector<double> diag(N);
        for(int i = 0; i < N; i++) diag[i] = T_diag[i] + V[i];

        // c. find eigenvalues
        std::vector<double> eigenvalues = find_eigenvalues(diag, e);

        // d. for each eigenvalue → inverse_iteration
        for(int k = 0; k < K; k++){
            all_eigenvalues.push_back(eigenvalues[k]);

            std::vector<double> vec = inverse_iteration(diag, e, eigenvalues[k]);
            all_eigenvectors.insert(all_eigenvectors.end(), vec.begin(), vec.end());
        }
    }

    // 5. save results
    cnpy::npy_save("../data/eigenvalues.npy",
                    all_eigenvalues.data(),
                    {static_cast<size_t>(M), static_cast<size_t>(K)},
                    "w");
    cnpy::npy_save("../data/eigenvectors.npy",
                    all_eigenvectors.data(),
                    {static_cast<size_t>(M), static_cast<size_t>(K), static_cast<size_t>(N)},
                    "w");
    return 0;
}
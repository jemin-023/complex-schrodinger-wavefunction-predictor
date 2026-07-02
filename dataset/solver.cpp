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
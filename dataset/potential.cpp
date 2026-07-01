#include <vector>
#include "cnpy.h"
#include <cmath>
#include <random>

// ===== Same constants as generate.cpp =====
constexpr int N = 256;
constexpr double xmin = -5.0;
constexpr double xmax = 5.0;
const double dx = (xmax - xmin) / (N - 1);

constexpr int M = 10000; // How many potentials to generate

// Random Gaussian mixture potential
double gaussian(double A, double mu, double sigma, double x){
    double d = x - mu;
    return A * std::exp(-(d * d) / (2.0 * sigma * sigma));
}

// === Core function: a random V(X) generator ===
std::vector<double> random_potential(std::mt19937& rng){
    std::vector<double> V(N, 0.0);

    std::uniform_real_distribution<double> amp_dist(-3.0, 3.0);
    std::uniform_real_distribution<double> mu_dist(xmin, xmax);
    std::uniform_real_distribution<double> sig_dist(0.2, 1.5);
    std::uniform_int_distribution<int>     K_dist(1, 5);

    int K = K_dist(rng);

    for(int k = 0; k < K; k++){
        double A     = amp_dist(rng);
        double mu    = mu_dist(rng);
        double sigma = sig_dist(rng);
        for(int i = 0; i < N; i++){
            double x = xmin + i * dx;
            V[i] += gaussian(A, mu, sigma, x);
        }
    }
    return V;
}


int main(){
    std::mt19937 rng(18);

    std::vector<double> flat;
    flat.reserve(M * N);

    for(auto i = 0; i < M; i++){
        auto V = random_potential(rng);
        flat.insert(flat.end(), V.begin(), V.end());
    }
    cnpy::npy_save(
        "data/potentials.npy",
        flat.data(),
        {static_cast<size_t>(M), static_cast<size_t>(N)},
        "w"
    );
}   
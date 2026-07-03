// Units chosen such that ħ²/(2m) = 1. (for simplicity)

#include <vector>
#include <cnpy>

using Matrix = std::vector<std::vector<double>>;


// ===== Global Variables =====
constexpr int N = 256;
constexpr double xmin = -5.0;
constexpr double xmax = 5.0;

const double dx = (xmax - xmin)/(N - 1);

int main(){
    Matrix T(N, std::vector<double>(N, 0.0));

    double inv_dx2 = 1.0 / (dx * dx);

    for (int i = 0; i < N; i++){
        T[i][i] = -2.0 * inv_dx2;

        if (i > 0) T[i][i - 1] = inv_dx2;

        if (i < N - 1) T[i][i + 1] = inv_dx2;
    }

    // === Filling the flat matrix to save it ===
    std::vector<double> flat;

    flat.reserve(N * N);

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
        flat.push_back(T[i][j]);
        }
    }

    cnpy::npy_save(
        "../data/T.npy",
        flat.data(),
        {static_cast<size_t>(N), static_cast<size_t>(N)},
        "w"
    );
    return 0;
}
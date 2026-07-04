import os
import numpy as np
import scipy.linalg
from tqdm import tqdm

N = 256
xmin = -5.0
xmax = 5.0
dx = (xmax - xmin) / (N - 1)
inv_dx2 = 1.0 / (dx ** 2)
M = 50000
K_eig = 18

def main():
    print("Generating T matrix...")
    T = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        T[i, i] = -2.0 * inv_dx2
        if i > 0:
            T[i, i - 1] = inv_dx2
        if i < N - 1:
            T[i, i + 1] = inv_dx2

    os.makedirs("data", exist_ok=True)
    np.save("data/T.npy", T)

    # Precompute X grid
    X = np.linspace(xmin, xmax, N)

    rng = np.random.default_rng(18)
    
    all_potentials = np.zeros((M, N), dtype=np.float64)
    all_eigenvalues = np.zeros((M, K_eig), dtype=np.float64)
    all_eigenvectors = np.zeros((M, K_eig, N), dtype=np.float64)

    # Off-diagonal elements are constant for all M potentials
    e_offdiag = np.full(N - 1, inv_dx2, dtype=np.float64)

    print(f"Generating {M} potentials and solving for eigenvalues/eigenvectors...")
    for m in tqdm(range(M)):
        # Generate random Gaussian mixture potential
        K_gauss = rng.integers(1, 6) # [1, 5] inclusive
        A = rng.uniform(-3.0, 3.0, K_gauss)
        mu = rng.uniform(xmin, xmax, K_gauss)
        sigma = rng.uniform(0.2, 1.5, K_gauss)
        
        V = np.zeros(N, dtype=np.float64)
        for k in range(K_gauss):
            d = X - mu[k]
            V += A[k] * np.exp(-(d ** 2) / (2.0 * sigma[k] ** 2))
        
        all_potentials[m] = V

        # Solve Hamiltonian
        # H is symmetric tridiagonal. Diagonal is T_diag + V.
        # Off-diagonal is e_offdiag.
        diag = -2.0 * inv_dx2 + V
        
        # We need the lowest K_eig eigenvalues
        # select='i' means by index. range is [0, K_eig-1]
        eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(
            diag, e_offdiag, select='i', select_range=(0, K_eig - 1)
        )
        
        all_eigenvalues[m] = eigvals
        # eigvecs returned by eigh_tridiagonal is of shape (N, K_eig)
        # We want (K_eig, N) so we transpose
        all_eigenvectors[m] = eigvecs.T

    print("Saving to disk...")
    np.save("data/potentials.npy", all_potentials)
    np.save("data/eigenvalues.npy", all_eigenvalues)
    np.save("data/eigenvectors.npy", all_eigenvectors)
    print("Dataset generation complete.")

if __name__ == "__main__":
    main()

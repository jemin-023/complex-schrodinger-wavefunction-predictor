# Dataset Generation

This directory contains the logic used to natively generate the potentials, kinetic energy matrices, eigenvalues, and eigenfunctions for training the Complex-Valued Neural Network.

## Overview
The dataset consists of 50,000 samples solving the 1D Time-Independent Schrödinger Equation (TISE):
$$ \hat{H} \psi_n(x) = E_n \psi_n(x) $$

Where the Hamiltonian operator $\hat{H}$ is defined as:
$$ \hat{H} = -\frac{1}{2}\nabla^2 + V(x) $$
*(using atomic units $\hbar = 1, m = 1$)*

## Algorithms

### 1. Kinetic Energy Matrix ($T$)
The kinetic energy operator is discretized using a finite-difference central approximation over $N=256$ spatial points with step size $\Delta x$. The codebase scales the Hamiltonian such that the effective kinetic energy matrix $T$ is given by:
$$ T_{i,i} = -\frac{2}{\Delta x^2} $$
$$ T_{i, i\pm1} = \frac{1}{\Delta x^2} $$

### 2. Random Potential Generation ($V(x)$)
Potentials are created as a random Gaussian mixture model with $K_{gauss} \in [1, 5]$ components:
$$ V(x) = \sum_{k=1}^{K_{gauss}} A_k \exp\left( - \frac{(x - \mu_k)^2}{2 \sigma_k^2} \right) $$
Where parameters are uniformly sampled:
- Amplitude $A_k \sim \mathcal{U}(-3.0, 3.0)$
- Mean $\mu_k \sim \mathcal{U}(-5.0, 5.0)$
- Standard Deviation $\sigma_k \sim \mathcal{U}(0.2, 1.5)$

### 3. Solving for Eigenstates
The discretized Hamiltonian matrix $H$ is a symmetric tridiagonal matrix defined by:
$$ H_{i,i} = T_{i,i} + V(x_i) $$
$$ H_{i, i\pm1} = T_{i, i\pm1} $$

The eigenvalues ($E_n$) and eigenvectors ($\psi_n$) are efficiently computed using SciPy's `scipy.linalg.eigh_tridiagonal` eigensolver, targeting the first $K=18$ lowest energy levels.

## How to generate
You can run the script to generate all 50,000 potentials directly:
```bash
python generate_data.py
```
Outputs are saved into the `data/` folder as `.npy` files.

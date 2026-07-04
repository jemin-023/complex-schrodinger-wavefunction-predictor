import time
import pickle
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal
from models.model import PsiNet
from dataset import load_data, train_val_split
import flax

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def main():
    print("Loading data...")
    try:
        V, E, psi = load_data()
        T_all = np.load("data/T.npy")
    except Exception as e:
        print("Could not load data:", e)
        return

    (V_train, E_train, psi_train), (V_val, E_val, psi_val) = train_val_split(V, E, psi)

    E_mean = float(jnp.mean(E_train))
    E_std  = float(jnp.std(E_train) + 1e-8)

    e_val = T_all[0, 1]
    T_diag = np.diag(T_all)
    
    num_samples = len(V_val)
    V_bench = V_val
    
    print(f"Benchmarking on {num_samples} validation samples...")
    
    # --- 1. Model Benchmark ---
    print("\n--- Model Benchmark ---")
    model = PsiNet(K=18)
    print("Loading model parameters from logs/checkpoint.pkl...")
    try:
        with open("logs/checkpoint.pkl", "rb") as f:
            params = pickle.load(f)
    except FileNotFoundError:
        print("logs/checkpoint.pkl not found! Falling back to random initialization.")
        rng = jax.random.PRNGKey(0)
        params = model.init(rng, jnp.zeros((1, 256)))
    
    param_count = count_params(params)
    print(f"Model Parameter Count: {param_count:,}")
    
    batch_size = 1024
    
    @jax.jit
    def predict_step(V_batch):
        return model.apply(params, V_batch)
    
    # Warmup
    print("Compiling model...")
    t0 = time.perf_counter()
    _ = predict_step(jnp.array(V_bench[:batch_size]))
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    print(f"Compilation time: {t1 - t0:.4f} seconds")
    
    # Timing model
    print("Timing model execution (batched)...")
    V_jax = jnp.array(V_bench)
    
    t0 = time.perf_counter()
    E_preds = []
    psi_preds = []
    for i in range(0, num_samples, batch_size):
        E_b, psi_b = predict_step(V_jax[i:i+batch_size])
        E_preds.append(E_b)
        psi_preds.append(psi_b)
    jax.block_until_ready(E_preds[-1])
    t1 = time.perf_counter()
    model_time = t1 - t0
    
    E_pred = jnp.concatenate(E_preds, axis=0)
    psi_pred = jnp.concatenate(psi_preds, axis=0)
    E_pred_denorm = E_pred * E_std + E_mean

    print(f"Model time for {num_samples} samples: {model_time:.4f} seconds")
    print(f"Model time per sample: {model_time / num_samples * 1000:.4f} ms")
    
    # --- 2. Actual Solver Benchmark (Python Scipy) ---
    print("\n--- Actual Solver Benchmark (Scipy eigh_tridiagonal) ---")
    off_diag = np.full(len(T_diag)-1, e_val)
    # Warmup
    _ = eigh_tridiagonal(T_diag + V_bench[0], off_diag, select='i', select_range=(0, 17))
    
    t0 = time.perf_counter()
    for i in range(num_samples):
        _ = eigh_tridiagonal(T_diag + V_bench[i], off_diag, select='i', select_range=(0, 17))
    t1 = time.perf_counter()
    solver_time = t1 - t0
    print(f"Solver time for {num_samples} samples: {solver_time:.4f} seconds")
    print(f"Solver time per sample: {solver_time / num_samples * 1000:.4f} ms")
    
    print(f"\nSpeedup: {solver_time / model_time:.2f}x")

    # --- 3. Accuracy Benchmark ---
    print("\n--- Accuracy Benchmark ---")
    E_true = np.array(E_val)
    psi_true = np.array(psi_val)
    
    mae_E = np.mean(np.abs(E_pred_denorm - E_true))
    rel_error_E = np.mean(np.abs(E_pred_denorm - E_true) / (np.abs(E_true) + 1e-8)) * 100
    
    # For wavefunctions, we compare probability densities because of the arbitrary global phase 
    # produced by standard eigensolvers
    density_pred = np.abs(psi_pred)**2
    density_true = np.abs(psi_true)**2
    mae_density = np.mean(np.abs(density_pred - density_true))
    
    print(f"Mean Absolute Error (Eigenvalues): {mae_E:.4f}")
    print(f"Mean Relative Error (Eigenvalues): {rel_error_E:.2f}%")
    print(f"Mean Absolute Error (|ψ|² Density): {mae_density:.6f}")

if __name__ == '__main__':
    main()

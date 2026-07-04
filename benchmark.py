import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigh_tridiagonal
from models.model import PsiNet
import flax

def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

def main():
    print("Loading data...")
    try:
        V_all = np.load("data/potentials.npy")
        T_all = np.load("data/T.npy")
    except Exception as e:
        print("Could not load data:", e)
        return

    e_val = T_all[0, 1]
    T_diag = np.diag(T_all)
    
    num_samples = 10000
    if len(V_all) < num_samples:
        num_samples = len(V_all)
        
    V_bench = V_all[:num_samples]
    
    print(f"Benchmarking on {num_samples} samples...")
    
    # --- 1. Model Benchmark ---
    print("\n--- Model Benchmark ---")
    model = PsiNet(K=18)
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.zeros((1, 256))
    params = model.init(rng, dummy_input)
    
    param_count = count_params(params)
    print(f"Model Parameter Count: {param_count:,}")
    
    # To optimize the model, we use jax.jit and batching.
    # We also evaluate in smaller batches if necessary, but 10,000 fits in memory usually.
    # Let's use a realistic batch size like 1024 to avoid OOM, or we can just JIT the batch step.
    batch_size = 1024
    
    @jax.jit
    def predict_step(V_batch):
        return model.apply(params, V_batch)
    
    # Warmup (compilation)
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
    for i in range(0, num_samples, batch_size):
        _ = predict_step(V_jax[i:i+batch_size])
    jax.block_until_ready(_)
    t1 = time.perf_counter()
    model_time = t1 - t0
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

if __name__ == '__main__':
    main()

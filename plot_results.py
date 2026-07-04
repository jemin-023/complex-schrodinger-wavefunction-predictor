import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from models.model import PsiNet
from dataset import load_data, train_val_split

def main():
    # 1. Load data
    V, E, psi = load_data()
    (V_train, E_train, psi_train), (V_val, E_val, psi_val) = train_val_split(V, E, psi)
    
    # Calculate normalization constants (same as training)
    E_mean = float(jnp.mean(E_train))
    E_std  = float(jnp.std(E_train) + 1e-8)
    
    # 2. Load model
    print("Loading model...")
    with open("logs/checkpoint.pkl", "rb") as f:
        params = pickle.load(f)
        
    model = PsiNet(K=18)
    
    # Helper to predict in batches to prevent GPU OOM
    def predict_in_batches(V_data, batch_size=32):
        n_samples = V_data.shape[0]
        E_preds = []
        psi_preds = []
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            V_b = V_data[start:end]
            E_b, psi_b = model.apply(params, V_b)
            # Evaluate JAX arrays to prevent accumulating trace/graphs
            E_preds.append(E_b)
            psi_preds.append(psi_b)
        return jnp.concatenate(E_preds, axis=0), jnp.concatenate(psi_preds, axis=0)
    
    # 3. Create plots directory
    os.makedirs("plots", exist_ok=True)
    
    # 4. Generate predictions at original resolution (256 points) for direct data comparison
    print("Evaluating model at 256 points in batches...")
    E_pred_256, psi_pred_256 = predict_in_batches(V_val, batch_size=32)
    E_pred_256_denorm = E_pred_256 * E_std + E_mean
    
    # Convert all to numpy arrays for plotting
    E_true = np.array(E_val)
    psi_true = np.array(psi_val)
    V_val_256 = np.array(V_val)
    
    E_pred_256_denorm = np.array(E_pred_256_denorm)
    psi_pred_256 = np.array(psi_pred_256)
    
    # ==========================================
    # Plot 1 — Eigenvalue prediction accuracy
    # ==========================================
    print("Generating Plot 1: Eigenvalue scatter...")
    plt.figure(figsize=(8, 8))
    plt.scatter(E_true.flatten(), E_pred_256_denorm.flatten(), alpha=0.3, s=5)
    plt.plot([E_true.min(), E_true.max()], 
             [E_true.min(), E_true.max()], 'r--', label='perfect')
    plt.xlabel("True Eₙ")
    plt.ylabel("Predicted Eₙ")
    plt.title("Eigenvalue Prediction")
    plt.legend()
    plt.savefig("plots/eigenvalue_scatter.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Plot 2 — Wavefunction visual comparison
    # ==========================================
    print("Generating Plot 2: Wavefunction visual comparison...")
    x_grid = np.linspace(-5, 5, 256)
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for idx, ax_row in enumerate(axes):
        # plot V(x)
        ax_row[0].plot(x_grid, V_val_256[idx])
        ax_row[0].set_title(f"Sample {idx} — V(x)")

        # plot ψ₀
        ax_row[1].plot(x_grid, np.abs(psi_true[idx, 0]), label='true')
        ax_row[1].plot(x_grid, np.abs(psi_pred_256[idx, 0]), '--', label='pred')
        ax_row[1].set_title("ψ₀ magnitude")
        ax_row[1].legend()

        # plot ψ₁
        ax_row[2].plot(x_grid, np.abs(psi_true[idx, 1]), label='true')
        ax_row[2].plot(x_grid, np.abs(psi_pred_256[idx, 1]), '--', label='pred')
        ax_row[2].set_title("ψ₁ magnitude")
        ax_row[2].legend()

    plt.tight_layout()
    plt.savefig("plots/wavefunction_comparison.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Plot 3 — Relative eigenvalue error per n
    # ==========================================
    print("Generating Plot 3: Relative error per quantum number...")
    K = 18
    rel_error = np.abs(E_pred_256_denorm - E_true) / (np.abs(E_true) + 1e-8)
    mean_error = rel_error.mean(axis=0)  # (K,)

    plt.figure(figsize=(8, 5))
    plt.bar(range(K), mean_error)
    plt.xlabel("Eigenvalue index n")
    plt.ylabel("Mean relative error")
    plt.title("Error vs quantum number")
    plt.savefig("plots/eigenvalue_error_per_n.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Plot 4 — Orthogonality check
    # ==========================================
    print("Generating Plot 4: Orthogonality check (256 grid)...")
    # pick one sample, compute gram matrix
    psi_sample = psi_pred_256[0]  # (K, 256)
    G = psi_sample @ np.conj(psi_sample).T  # (K, K)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.abs(G), cmap='viridis')
    plt.colorbar()
    plt.title("Gram matrix |⟨ψₙ|ψₘ⟩| — should be identity (256 grid)")
    plt.savefig("plots/orthogonality.png", dpi=400, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Heatmap 1 — Probability density |ψ_n(x)|^2
    # ==========================================
    print("Generating Heatmap 1: Probability density...")
    plt.figure(figsize=(10, 6))
    prob_density = np.abs(psi_pred_256[0])**2
    plt.imshow(prob_density, aspect='auto', extent=[-5, 5, K-0.5, -0.5], cmap='magma')
    plt.colorbar(label='|ψₙ(x)|²')
    plt.xlabel('x')
    plt.ylabel('Eigenstate n')
    plt.title('Probability Density Map')
    plt.savefig('plots/heatmap_probability_density.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Heatmap 2 — Error map |ψ_n_pred - ψ_n_true|
    # ==========================================
    print("Generating Heatmap 2: Error map...")
    plt.figure(figsize=(10, 6))
    error_map = np.abs(psi_pred_256[0] - psi_true[0])
    plt.imshow(error_map, aspect='auto', extent=[-5, 5, K-0.5, -0.5], cmap='Reds')
    plt.colorbar(label='Absolute Error')
    plt.xlabel('x')
    plt.ylabel('Eigenstate n')
    plt.title('Absolute Error Map')
    plt.savefig('plots/heatmap_error_map.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Heatmap 3 — Phase map arg(ψ_n(x))
    # ==========================================
    print("Generating Heatmap 3: Phase map...")
    plt.figure(figsize=(10, 6))
    phase_map = np.angle(psi_pred_256[0])
    plt.imshow(phase_map, aspect='auto', extent=[-5, 5, K-0.5, -0.5], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    plt.colorbar(label='Phase arg(ψₙ)')
    plt.xlabel('x')
    plt.ylabel('Eigenstate n')
    plt.title('Phase Map')
    plt.savefig('plots/heatmap_phase_map.png', dpi=400, bbox_inches='tight')
    plt.close()
    
    # ==========================================
    # Heatmap 4 — Average Probability Density Error (Across all val samples)
    # ==========================================
    print("Generating Heatmap 4: Average Probability Density Error...")
    plt.figure(figsize=(10, 6))
    # Calculate difference in probability densities (to avoid global phase issues)
    density_pred = np.abs(psi_pred_256)**2
    density_true = np.abs(psi_true)**2
    avg_density_error = np.mean(np.abs(density_pred - density_true), axis=0) # shape: (K, 256)
    
    plt.imshow(avg_density_error, aspect='auto', extent=[-5, 5, K-0.5, -0.5], cmap='inferno')
    plt.colorbar(label='Mean Absolute Error in |ψ|²')
    plt.xlabel('x')
    plt.ylabel('Eigenstate n')
    plt.title('Average Probability Density Error (Validation Set)\nLower is better')
    plt.savefig('plots/heatmap_avg_density_error.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Heatmap 5 — 2D Histogram of Eigenvalues (Density map of predictions)
    # ==========================================
    print("Generating Heatmap 5: 2D Histogram of Eigenvalues...")
    plt.figure(figsize=(8, 8))
    plt.hist2d(E_true.flatten(), E_pred_256_denorm.flatten(), bins=100, cmap='plasma', cmin=1)
    plt.plot([E_true.min(), E_true.max()], [E_true.min(), E_true.max()], 'r--', alpha=0.8, label='Perfect Prediction')
    plt.colorbar(label='Count / Density')
    plt.xlabel("True Eₙ")
    plt.ylabel("Predicted Eₙ")
    plt.title("Eigenvalue Prediction Density (Heatmap)")
    plt.legend()
    plt.savefig('plots/heatmap_eigenvalue_density.png', dpi=400, bbox_inches='tight')
    plt.close()

    # ==========================================
    # Heatmap 6 — Average Orthogonality (Gram Matrix) across all samples
    # ==========================================
    print("Generating Heatmap 6: Average Orthogonality Matrix...")
    # Compute Gram matrix for all samples using batched matrix multiplication
    # psi_pred_256 is (M, K, 256). We want (M, K, K)
    G_all = np.einsum('mij,mkj->mik', psi_pred_256, np.conj(psi_pred_256))
    avg_G = np.mean(np.abs(G_all), axis=0) # shape: (K, K)
    
    plt.figure(figsize=(7, 6))
    plt.imshow(avg_G, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Mean |⟨ψₙ|ψₘ⟩|')
    plt.title("Average Gram Matrix (Validation Set)\nShould perfectly match Identity")
    plt.xlabel('Eigenstate m')
    plt.ylabel('Eigenstate n')
    plt.savefig('plots/heatmap_avg_orthogonality.png', dpi=400, bbox_inches='tight')
    plt.close()

    print("All main plots generated and saved successfully in the 'plots/' directory!")

if __name__ == "__main__":
    main()

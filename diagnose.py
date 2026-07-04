"""Diagnostic script to understand the training plateau."""
import numpy as np
import jax
import jax.numpy as jnp

# Load raw data
V = np.load("data/potentials.npy")
E = np.load("data/eigenvalues.npy")
psi = np.load("data/eigenvectors.npy")

print("=== Data shapes ===")
print(f"V:   {V.shape}  dtype={V.dtype}")
print(f"E:   {E.shape}  dtype={E.dtype}")
print(f"psi: {psi.shape}  dtype={psi.dtype}")

print(f"\n=== Value ranges ===")
print(f"V:   min={V.min():.4f}  max={V.max():.4f}  mean={V.mean():.4f}")
print(f"E:   min={E.min():.4f}  max={E.max():.4f}  mean={E.mean():.4f}")
print(f"psi real: min={psi.real.min():.6f}  max={psi.real.max():.6f}")
print(f"psi imag: min={psi.imag.min():.6f}  max={psi.imag.max():.6f}")
print(f"psi |.|:  min={np.abs(psi).min():.6f}  max={np.abs(psi).max():.6f}")

print(f"\n=== Normalization check (first 5 samples, first 3 modes) ===")
for i in range(5):
    for k in range(3):
        norm_sq = np.sum(np.abs(psi[i, k]) ** 2)
        print(f"  sample {i}, mode {k}: sum|psi|^2 = {norm_sq:.6f}")

print(f"\n=== Are ground truth psi purely real? ===")
imag_frac = np.mean(np.abs(psi.imag)) / (np.mean(np.abs(psi)) + 1e-10)
print(f"  mean(|imag|) / mean(|psi|) = {imag_frac:.6e}")
print(f"  max(|imag|) = {np.abs(psi.imag).max():.6e}")
if imag_frac < 1e-6:
    print("  ==> Ground truth wavefunctions are PURELY REAL")

print(f"\n=== Phase alignment test ===")
# Simulate what loss_psi does: take two normalized vectors and check the loss
psi_j = jnp.asarray(psi)

# Random prediction baseline
key = jax.random.PRNGKey(42)
rand_psi = jax.random.normal(key, shape=psi_j[:10].shape) + 0j
# Normalize
rand_norm = jnp.sqrt((jnp.abs(rand_psi)**2).sum(axis=-1, keepdims=True) + 1e-12)
rand_psi = rand_psi / rand_norm

# True prediction (should give ~0)
overlap_true = jnp.sum(jnp.conj(psi_j[:10]) * psi_j[:10], axis=-1, keepdims=True)
phase_true = overlap_true / (jnp.abs(overlap_true) + 1e-8)
aligned_true = psi_j[:10] * jnp.conj(phase_true)
loss_true = jnp.mean(jnp.abs(aligned_true - psi_j[:10])**2) / (jnp.mean(jnp.abs(psi_j[:10])**2) + 1e-8)
print(f"  loss_psi(true, true)   = {float(loss_true):.6f}  (should be ~0)")

# Random prediction
overlap_rand = jnp.sum(jnp.conj(psi_j[:10]) * rand_psi, axis=-1, keepdims=True)
phase_rand = overlap_rand / (jnp.abs(overlap_rand) + 1e-8)
aligned_rand = rand_psi * jnp.conj(phase_rand)
loss_rand = jnp.mean(jnp.abs(aligned_rand - psi_j[:10])**2) / (jnp.mean(jnp.abs(psi_j[:10])**2) + 1e-8)
print(f"  loss_psi(random, true) = {float(loss_rand):.6f}  (should be ~2)")

# Sign-flipped prediction (should give ~0 after alignment)
flipped_psi = -psi_j[:10]
overlap_flip = jnp.sum(jnp.conj(psi_j[:10]) * flipped_psi, axis=-1, keepdims=True)
phase_flip = overlap_flip / (jnp.abs(overlap_flip) + 1e-8)
aligned_flip = flipped_psi * jnp.conj(phase_flip)
loss_flip = jnp.mean(jnp.abs(aligned_flip - psi_j[:10])**2) / (jnp.mean(jnp.abs(psi_j[:10])**2) + 1e-8)
print(f"  loss_psi(-true, true)  = {float(loss_flip):.6f}  (should be ~0 if phase align works)")

print(f"\n=== Per-mode difficulty ===")
# Check mean |psi|^2 per mode (higher modes may have smaller magnitudes)
for k in range(min(18, psi.shape[1])):
    mean_mag = np.mean(np.abs(psi[:, k])**2)
    max_val = np.abs(psi[:, k]).max()
    print(f"  mode {k:2d}: mean|psi|^2={mean_mag:.6f}  max|psi|={max_val:.4f}")

print(f"\n=== Eigenvalue ordering check ===")
# Are eigenvalues sorted per sample?
for i in range(5):
    diffs = np.diff(E[i])
    sorted_check = np.all(diffs >= 0)
    print(f"  sample {i}: E = [{E[i,0]:.2f}, ..., {E[i,-1]:.2f}]  sorted={sorted_check}")

print(f"\n=== Model output check (untrained) ===")
from models.model import PsiNet
model = PsiNet(K=18)
key = jax.random.PRNGKey(0)
dummy_V = jnp.ones((2, 256))
params = model.init(key, dummy_V)
E_out, psi_out = model.apply(params, dummy_V)
print(f"  E_out shape: {E_out.shape}, dtype: {E_out.dtype}")
print(f"  psi_out shape: {psi_out.shape}, dtype: {psi_out.dtype}")
print(f"  psi_out |.|: min={float(jnp.abs(psi_out).min()):.6f}  max={float(jnp.abs(psi_out).max()):.6f}")
print(f"  psi_out norm check: sum|psi|^2 per mode = {float((jnp.abs(psi_out[0,0])**2).sum()):.6f}")

# Check gradient flow
print(f"\n=== Gradient flow check ===")
from losses import loss_psi, loss_eigenvalue, total_loss

V_b = jnp.asarray(V[:4])
E_b = jnp.asarray(E[:4])
psi_b = jnp.asarray(psi[:4])
E_mean = jnp.mean(jnp.asarray(E))
E_std = jnp.std(jnp.asarray(E)) + 1e-8
E_b_norm = (E_b - E_mean) / E_std

def loss_fn(params):
    E_pred, psi_pred = model.apply(params, V_b)
    total, (L_E, L_psi, L_phys, L_orth) = total_loss(
        E_pred, psi_pred, E_b_norm, psi_b, V_b, 10.0/255, E_mean, E_std
    )
    return total, (L_E, L_psi, L_phys, L_orth)

(total, (L_E, L_psi, L_phys, L_orth)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
print(f"  Initial losses: total={float(total):.4f} E={float(L_E):.4f} psi={float(L_psi):.4f} phys={float(L_phys):.4f} orth={float(L_orth):.4f}")

# Check gradient norms per layer
grad_norms = jax.tree.map(lambda x: float(jnp.sqrt(jnp.sum(x**2))), grads)
flat_grads = jax.tree.leaves(grad_norms)
print(f"  Gradient norms: min={min(flat_grads):.6f}  max={max(flat_grads):.6f}  mean={sum(flat_grads)/len(flat_grads):.6f}")
print(f"  Any NaN grads: {any(jnp.isnan(x).any() for x in jax.tree.leaves(grads))}")

# Separate gradient contributions
def loss_psi_only(params):
    E_pred, psi_pred = model.apply(params, V_b)
    return loss_psi(psi_pred, psi_b)

def loss_E_only(params):
    E_pred, psi_pred = model.apply(params, V_b)
    return loss_eigenvalue(E_pred, E_b_norm)

grads_psi = jax.grad(loss_psi_only)(params)
grads_E = jax.grad(loss_E_only)(params)

# Check grads specifically in the eigenfunction head vs shared backbone
psi_grad_norms = jax.tree.map(lambda x: float(jnp.sqrt(jnp.sum(x**2))), grads_psi)
E_grad_norms = jax.tree.map(lambda x: float(jnp.sqrt(jnp.sum(x**2))), grads_E)

print(f"\n=== Layer-by-layer gradient analysis ===")
flat_params = dict(jax.tree.leaves_with_path(grads))
flat_psi = dict(jax.tree.leaves_with_path(grads_psi))
flat_E = dict(jax.tree.leaves_with_path(grads_E))

for path in sorted(flat_params.keys(), key=str):
    p_norm = float(jnp.sqrt(jnp.sum(flat_params[path]**2)))
    psi_n = float(jnp.sqrt(jnp.sum(flat_psi[path]**2)))
    e_n = float(jnp.sqrt(jnp.sum(flat_E[path]**2)))
    name = '/'.join(str(k) for k in path)
    if p_norm > 0.001:  # only show significant
        print(f"  {name:60s}  total={p_norm:.4f}  psi={psi_n:.4f}  E={e_n:.4f}")

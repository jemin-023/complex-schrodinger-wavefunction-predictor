import jax.numpy as jnp

dx = 10.0 / 255
N = 256
K = 18

# E_pred: (batch, K) real  — normalized (zero-mean, unit-var)
# E_true: (batch, K) real  — normalized (zero-mean, unit-var)
def loss_eigenvalue(E_pred, E_true):
    return jnp.mean((E_pred - E_true) ** 2)

def loss_ortho(psi_pred):
    # Overlap matrix: shape (batch, K, K)
    G = psi_pred @ jnp.conj(psi_pred).swapaxes(-1, -2)
    mask = 1.0 - jnp.eye(K)
    loss = jnp.mean(jnp.abs(G) ** 2 * mask)
    return loss

def loss_psi(psi_pred, psi_true):
    # Align phases to handle U(1) symmetry
    overlap = jnp.sum(jnp.conj(psi_true) * psi_pred, axis=-1, keepdims=True)
    phase = overlap / (jnp.abs(overlap) + 1e-8)
    psi_aligned = psi_pred * jnp.conj(phase)
    
    # Scale relative to the magnitude of the ground truth eigenfunctions
    # to ensure comparable gradient scale to eigenvalue loss.
    # Orthogonal states = 2.0, Aligned states = 0.0
    numerator = jnp.mean(jnp.abs(psi_aligned - psi_true) ** 2)
    denominator = jnp.mean(jnp.abs(psi_true) ** 2) + 1e-8
    loss = numerator / denominator
    return loss

# psi_pred: (batch, K, 256) complex
# E_pred:   (batch, K) real  — normalized
# V:        (batch, 256) real
# E_mean, E_std: scalars for denormalization (constants, no grad)
def loss_physics(psi_pred, E_pred, V, dx, E_mean, E_std):
    # Denormalize eigenvalues back to physical units
    E_actual = E_pred * E_std + E_mean
    # kinetic term
    psi_interior = psi_pred[:, :, 1:-1]
    psi_left     = psi_pred[:, :, :-2]
    psi_right    = psi_pred[:, :, 2:]
    kinetic      = (psi_right - 2*psi_interior + psi_left) / dx**2
    # potential term
    potential = V[:, 1:-1].astype(jnp.complex64)[:, None, :] * psi_interior
    # full H*psi
    H_psi = kinetic + potential
    # residual: Hψ - Eψ (interior only)
    residual = H_psi - E_actual[:, :, None].astype(jnp.complex64) * psi_interior
    # Normalize by |E_mean| (fixed constant — no gradient flows through scale)
    E_scale = jnp.abs(E_mean) + 1e-8
    loss = jnp.mean(jnp.abs(residual / E_scale) ** 2)
    return loss

def total_loss(E_pred, psi_pred, E_true, psi_true, V, dx, E_mean, E_std):
    # λ1 (E), λ2 (psi), λ3 (phys), λ4 (ortho)
    λ1, λ2, λ3, λ4 = 0.001, 10.0, 1000.0, 1.0

    L_E    = loss_eigenvalue(E_pred, E_true)
    L_psi  = loss_psi(psi_pred, psi_true)
    L_phys = loss_physics(psi_pred, E_pred, V, dx, E_mean, E_std)
    L_orth = loss_ortho(psi_pred)

    total = λ1*L_E + λ2*L_psi + λ3*L_phys + λ4*L_orth

    # return total AND all 4 individually
    return total, (L_E, L_psi, L_phys, L_orth)
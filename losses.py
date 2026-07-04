import jax.numpy as jnp

dx = 10.0 / 255
N = 256
K = 18


# E_pred: (batch, K) real
# E_true: (batch, K) real
def loss_eigenvalue(E_pred, E_true):
    return jnp.mean((E_pred - E_true) ** 2)


# psi_pred: (batch, K, 256) complex
def loss_norm(psi_pred):
    norm = jnp.sum(jnp.abs(psi_pred)**2, axis=-1) * dx
    loss = jnp.mean((norm - 1.0)**2)
    return loss

def loss_ortho(psi_pred):
    G = psi_pred @ jnp.conj(psi_pred).swapaxes(-1, -2) * dx
    mask = 1.0 - jnp.eye(K)
    loss = jnp.mean(jnp.abs(G) ** 2 * mask)
    return loss

def loss_psi(psi_pred, psi_true):
    overlap = jnp.sum(jnp.conj(psi_true) * psi_pred, axis=-1, keepdims=True)
    phase = overlap / (jnp.abs(overlap) + 1e-8)
    psi_aligned = psi_pred * jnp.conj(phase)
    loss = jnp.mean(jnp.abs(psi_aligned - psi_true) ** 2)
    return loss

# psi_pred: (batch, K, 256) complex
# E_pred:   (batch, K) real
# V:        (batch, 256) real

def loss_physics(psi_pred, E_pred, V, dx):
    # kinetic term
    psi_interior = psi_pred[:, :, 1:-1]
    psi_left     = psi_pred[:, :, :-2]
    psi_right    = psi_pred[:, :, 2:]
    kinetic      = -(psi_right - 2*psi_interior + psi_left) / dx**2
    # potential term
    potential = V[:, 1:-1].astype(jnp.complex64)[:, None, :] * psi_interior
    # full H*psi
    H_psi = kinetic + potential
    # residual: Hψ - Eψ (interior only)
    residual = H_psi - E_pred[:, :, None].astype(jnp.complex64) * psi_interior
    loss = jnp.mean(jnp.abs(residual) ** 2)
    return loss

def total_loss(E_pred, psi_pred, E_true, psi_true, V, dx):
    # weights
    λ1 = 1.0  # eigenvalue
    λ2 = 1.0  # psi
    λ3 = 0.1  # physics
    λ4 = 0.1  # norm
    λ5 = 0.01  # ortho
    
    L_E    = loss_eigenvalue(E_pred, E_true)
    L_psi  = loss_psi(psi_pred, psi_true)
    L_phys = loss_physics(psi_pred, E_pred, V, dx)
    L_norm = loss_norm(psi_pred)
    L_orth = loss_ortho(psi_pred)
    
    return λ1*L_E + λ2*L_psi + λ3*L_phys + λ4*L_norm + λ5*L_orth
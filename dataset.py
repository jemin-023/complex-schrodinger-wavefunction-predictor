import numpy as np
import jax
import jax.numpy as jnp
from typing import Iterator


def load_data(data_dir: str = "data"):
    V = np.load(f"{data_dir}/potentials.npy")
    E = np.load(f"{data_dir}/eigenvalues.npy")
    psi = np.load(f"{data_dir}/eigenvectors.npy")

    return (
        jnp.asarray(V),
        jnp.asarray(E),
        jnp.asarray(psi),
    )


def train_val_split(V, E, psi, val_ratio=0.1):
    N = len(V)
    split = int(N * (1 - val_ratio))

    V_train, V_val = V[:split], V[split:]
    E_train, E_val = E[:split], E[split:]
    psi_train, psi_val = psi[:split], psi[split:]

    return (
        (V_train, E_train, psi_train),
        (V_val, E_val, psi_val),
    )


def dataloader(V, E, psi, batch_size, key) -> Iterator:
    N = len(V)

    indices = jax.random.permutation(key, N)

    for start in range(0, N, batch_size):
        batch_idx = indices[start:start + batch_size]

        yield (
            V[batch_idx],
            E[batch_idx],
            psi[batch_idx],
        )
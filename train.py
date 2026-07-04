import csv
import os
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from models.model import PsiNet
from dataset import load_data, train_val_split, dataloader
from losses import total_loss

N          = 256
dx         = 10.0 / 255
K          = 18
BATCH_SIZE = 32
EPOCHS     = 200
LR         = 1e-3

def create_train_state(key, learning_rate, total_steps):
    model   = PsiNet(K=K)
    dummy_V = jnp.ones((1, N))
    params  = model.init(key, dummy_V)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=total_steps // 10,
        decay_steps=total_steps,
        end_value=learning_rate * 0.01,
    )
    tx      = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adam(schedule),
    )
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

@jax.jit
def train_step(state, V_batch, E_batch, psi_batch, E_mean, E_std):
    def loss_fn(params):
        E_pred, psi_pred = state.apply_fn(params, V_batch)
        total, components = total_loss(E_pred, psi_pred, E_batch, psi_batch, V_batch, dx, E_mean, E_std)
        return total, components
    (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, components

@jax.jit
def val_step(state, V_batch, E_batch, psi_batch, E_mean, E_std):
    E_pred, psi_pred = state.apply_fn(state.params, V_batch)
    total, components = total_loss(E_pred, psi_pred, E_batch, psi_batch, V_batch, dx, E_mean, E_std)
    return total, components

def main():
    V, E, psi = load_data()
    (V_train, E_train, psi_train), (V_val, E_val, psi_val) = train_val_split(V, E, psi)

    # Normalize eigenvalues to zero-mean, unit-variance
    E_mean = jnp.mean(E_train)
    E_std  = jnp.std(E_train) + 1e-8
    E_train = (E_train - E_mean) / E_std
    E_val   = (E_val   - E_mean) / E_std
    print(f"E normalization: mean={float(E_mean):.2f}, std={float(E_std):.2f}")

    # Compute schedule parameters
    steps_per_epoch = len(V_train) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    print(f"Training: {EPOCHS} epochs × {steps_per_epoch} steps = {total_steps} total steps")

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, LR, total_steps)

    # CSV logging
    log_path = "logs/training_log.csv"
    os.makedirs("logs", exist_ok=True)

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_total', 'L_E', 'L_psi', 'L_phys', 'L_orth', 'val_total'])

        for epoch in range(EPOCHS):
            key, subkey = jax.random.split(key)
            train_loss  = 0.0
            train_comps = jnp.zeros(4)
            n_batches   = 0

            for V_b, E_b, psi_b in dataloader(V_train, E_train, psi_train, BATCH_SIZE, subkey):
                state, loss, (L_E, L_psi, L_phys, L_orth) = train_step(
                    state, V_b, E_b, psi_b, E_mean, E_std
                )
                train_loss  += loss
                train_comps += jnp.array([L_E, L_psi, L_phys, L_orth])
                n_batches   += 1

            train_loss  /= n_batches
            train_comps /= n_batches

            val_loss, (vL_E, vL_psi, vL_phys, vL_orth) = val_step(
                state, V_val[:BATCH_SIZE], E_val[:BATCH_SIZE], psi_val[:BATCH_SIZE],
                E_mean, E_std
            )

            print(
                f"Epoch {epoch+1:03d} | "
                f"total {train_loss:.4f} | "
                f"E {train_comps[0]:.4f} | "
                f"ψ {train_comps[1]:.4f} | "
                f"phys {train_comps[2]:.4f} | "
                f"orth {train_comps[3]:.4f} | "
                f"val {val_loss:.4f}"
            )

            # Save row after each epoch
            writer.writerow([
                epoch+1,
                float(train_loss),
                float(train_comps[0]),
                float(train_comps[1]),
                float(train_comps[2]),
                float(train_comps[3]),
                float(val_loss)
            ])
            f.flush()  # write immediately, don't buffer

    # Save final parameters
    import pickle
    with open("logs/checkpoint.pkl", "wb") as pf:
        pickle.dump(state.params, pf)
    print("Saved final checkpoint to logs/checkpoint.pkl")

if __name__ == "__main__":
    main()
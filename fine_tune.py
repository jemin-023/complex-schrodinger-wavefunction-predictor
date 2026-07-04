import csv
import os
import pickle
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from models.model import PsiNet
from dataset import load_data, train_val_split, dataloader
from losses import loss_eigenvalue, loss_psi, loss_physics, loss_ortho

# Constants
N_colloc   = 512
dx_colloc  = 10.0 / (N_colloc - 1)
K          = 18
BATCH_SIZE = 32
EPOCHS     = 50
LR         = 1e-4  # Smaller learning rate for fine-tuning

# Adjusted Loss Weights for Fine-Tuning
# We increase λ3 (PDE residual) from 0.1 to 2.0 and λ4 (orthogonality) from 0.1 to 0.5
# We keep λ2 (supervised wavefunction loss) at 10.0 to act as regularization and prevent weight distortion
λ1, λ2, λ3, λ4 = 0.001, 10.0, 2.0, 0.5

def total_loss_finetune(E_pred, psi_pred, E_true, psi_true, V, dx, E_mean, E_std):
    L_E    = loss_eigenvalue(E_pred, E_true)
    L_psi  = loss_psi(psi_pred, psi_true)
    L_phys = loss_physics(psi_pred, E_pred, V, dx, E_mean, E_std)
    L_orth = loss_ortho(psi_pred)

    total = λ1*L_E + λ2*L_psi + λ3*L_phys + λ4*L_orth
    return total, (L_E, L_psi, L_phys, L_orth)

def create_train_state(key, learning_rate, total_steps, params):
    model   = PsiNet(K=K)
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
        total, components = total_loss_finetune(E_pred, psi_pred, E_batch, psi_batch, V_batch, dx_colloc, E_mean, E_std)
        return total, components
    (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, components

@jax.jit
def val_step(state, V_batch, E_batch, psi_batch, E_mean, E_std):
    E_pred, psi_pred = state.apply_fn(state.params, V_batch)
    total, components = total_loss_finetune(E_pred, psi_pred, E_batch, psi_batch, V_batch, dx_colloc, E_mean, E_std)
    return total, components

def main():
    # 1. Load checkpoint
    checkpoint_path = "checkpoint.pkl"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found! Run the save_model.sh script first.")
        
    print(f"Loading pre-trained parameters from {checkpoint_path}...")
    with open(checkpoint_path, 'rb') as f:
        params = pickle.load(f)

    # 2. Load data and split
    V, E, psi = load_data()
    (V_train, E_train, psi_train), (V_val, E_val, psi_val) = train_val_split(V, E, psi)

    # 3. Normalize eigenvalues
    E_mean = jnp.mean(E_train)
    E_std  = jnp.std(E_train) + 1e-8
    E_train = (E_train - E_mean) / E_std
    E_val   = (E_val   - E_mean) / E_std

    # 4. Upsample data to 512 points using cubic interpolation
    print(f"Upsampling input potentials and wavefunctions to {N_colloc} points...")
    
    # Upsample potentials (batch, 256) -> (batch, 512)
    V_train_colloc = jax.image.resize(V_train, (V_train.shape[0], N_colloc), method="cubic")
    V_val_colloc   = jax.image.resize(V_val, (V_val.shape[0], N_colloc), method="cubic")
    
    # Upsample wavefunctions (batch, K, 256) -> (batch, K, 512)
    psi_train_colloc = jax.image.resize(psi_train, (psi_train.shape[0], K, N_colloc), method="cubic")
    psi_val_colloc   = jax.image.resize(psi_val, (psi_val.shape[0], K, N_colloc), method="cubic")

    # Compute schedule parameters
    steps_per_epoch = len(V_train_colloc) // BATCH_SIZE
    total_steps = EPOCHS * steps_per_epoch
    print(f"Fine-tuning: {EPOCHS} epochs × {steps_per_epoch} steps = {total_steps} total steps")

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    state = create_train_state(init_key, LR, total_steps, params)

    # Logging setup
    log_path = "logs/fine_tune_log.csv"
    os.makedirs("logs", exist_ok=True)

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_total', 'L_E', 'L_psi', 'L_phys', 'L_orth', 'val_total'])

        for epoch in range(EPOCHS):
            key, subkey = jax.random.split(key)
            train_loss  = 0.0
            train_comps = jnp.zeros(4)
            n_batches   = 0

            for V_b, E_b, psi_b in dataloader(V_train_colloc, E_train, psi_train_colloc, BATCH_SIZE, subkey):
                state, loss, (L_E, L_psi, L_phys, L_orth) = train_step(
                    state, V_b, E_b, psi_b, E_mean, E_std
                )
                train_loss  += loss
                train_comps += jnp.array([L_E, L_psi, L_phys, L_orth])
                n_batches   += 1

            train_loss  /= n_batches
            train_comps /= n_batches

            val_loss, (vL_E, vL_psi, vL_phys, vL_orth) = val_step(
                state, V_val_colloc[:BATCH_SIZE], E_val[:BATCH_SIZE], psi_val_colloc[:BATCH_SIZE],
                E_mean, E_std
            )

            print(
                f"Epoch {epoch+1:02d} | "
                f"total {train_loss:.4f} | "
                f"E {train_comps[0]:.4f} | "
                f"ψ {train_comps[1]:.4f} | "
                f"phys {train_comps[2]:.4f} | "
                f"orth {train_comps[3]:.4f} | "
                f"val {val_loss:.4f}"
            )

            writer.writerow([
                epoch+1,
                float(train_loss),
                float(train_comps[0]),
                float(train_comps[1]),
                float(train_comps[2]),
                float(train_comps[3]),
                float(val_loss)
            ])
            f.flush()

    # Save final fine-tuned parameters
    with open("fine_tuned_checkpoint.pkl", "wb") as f:
        pickle.dump(state.params, f)
    print("Fine-tuning completed. Saved fine-tuned parameters to fine_tuned_checkpoint.pkl.")

if __name__ == "__main__":
    main()

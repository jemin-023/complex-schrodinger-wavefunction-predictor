import typing_extensions
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class ComplexLinear(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):   # x: complex array

        W_real = nn.Dense(self.features, name='W_real')
        W_imag = nn.Dense(self.features, name='W_imag')

        real = W_real(x.real) - W_imag(x.imag)
        imag = W_real(x.imag) + W_imag(x.real)

        return real + 1j * imag

class ModRelu(nn.Module):

    @nn.compact
    def __call__(self, z):

        b = self.param('b', nn.initializers.zeros, (z.shape[-1],))

        magnitude = jnp.sqrt((z.real * z.real) + (z.imag * z.imag) + 1e-12)

        scale = jax.nn.relu(magnitude + b) / (magnitude + 1e-8)

        return z*scale

class ComplexConv1d(nn.Module):
    features: int
    kernel_size: int
    strides: int = 1

    @nn.compact
    def __call__(self, x):  # x: complex (batch, length, channels)

        W_real = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), strides=(self.strides,), padding='SAME', name='W_real')
        W_imag = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), strides=(self.strides,), padding='SAME', name='W_imag')

        real = W_real(x.real) - W_imag(x.imag)
        imag = W_real(x.imag) + W_imag(x.real)

        return real + 1j * imag

class PsiNet(nn.Module):
    K: int = 18

    @nn.compact
    def __call__(self, V):
        x = V.astype(jnp.complex64)    # (batch, 256)
        x = x[..., None]               # (batch, 256, 1)

        # Encoder with pooling (strided convolutions)
        x = ComplexConv1d(64, 5, strides=2)(x)  # 256 -> 128
        x = ModRelu()(x)
        x = ComplexConv1d(128, 5, strides=2)(x) # 128 -> 64
        x = ModRelu()(x)
        x = ComplexConv1d(128, 3, strides=2)(x) # 64 -> 32
        x = ModRelu()(x)
        x = ComplexConv1d(128, 3, strides=2)(x) # 32 -> 16
        x = ModRelu()(x)

        # ===== Eigenvalue head (needs global info) =====
        e = jnp.mean(x, axis=1)      # (batch, 128)
        e = ComplexLinear(128)(e)
        e = ModRelu()(e)
        eigvals = ComplexLinear(self.K)(e).real   # (batch, K)

        # ===== Eigenfunction head (Dense Decoder for global mapping) =====
        batch = x.shape[0]
        d = x.reshape((batch, -1))  # (batch, 16 * 128) = (batch, 2048)
        
        # Dense projection
        d = ComplexLinear(1024)(d)
        d = ModRelu()(d)
        d = ComplexLinear(self.K * 256)(d)
        
        # Reshape back to spatial structure
        p = d.reshape((batch, 256, self.K))  # (batch, 256, K)
        
        # Ground truth is purely real, so output real part
        p_real = p.real
        p_complex = p_real + 0j
        psi = p_complex.swapaxes(-1, -2)         # (batch, K, 256)
        norm = jnp.sqrt((jnp.abs(psi)**2).sum(axis=-1, keepdims=True) + 1e-12)
        psi = psi / norm

        return eigvals, psi
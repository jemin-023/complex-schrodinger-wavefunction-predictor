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

        b = self.param('b', nn.initializers.zeros, z.shape)

        magnitude = jnp.sqrt((z.real * z.real) + (z.imag * z.imag))

        scale = jax.nn.relu(magnitude + b) / (magnitude + 1e-8)

        return z*scale

class ComplexConv1d(nn.Module):
    features: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):  # x: complex (batch, length, channels)

        W_real = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), padding='SAME', name='W_real')
        W_imag = nn.Conv(features=self.features, kernel_size=(self.kernel_size,), padding='SAME', name='W_imag')

        real = W_real(x.real) - W_imag(x.imag)
        imag = W_real(x.imag) + W_imag(x.real)

        return real + 1j * imag

class ComplexAttention(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):

        Q = ComplexLinear(self.features)(x)
        K = ComplexLinear(self.features)(x)
        V = ComplexLinear(self.features)(x)

        scores = (Q.real @ K.real.swapaxes(-1, -2) + Q.imag @ K.imag.swapaxes(-1, -2))
        scores = scores / jnp.sqrt(self.features)

        attn = jax.nn.softmax(scores, axis=-1)
        
        real = attn @ V.real
        imag = attn @ V.imag

        out = real + 1j * imag

        return out

class PsiNet(nn.Module):

    K: int = 5

    @nn.compact
    def __call__(self, V):

        # (batch,256)
        x = V.astype(jnp.complex64)

        # (batch,256,1)
        x = x[...,None]

        x = ComplexConv1d(64,3)(x)
        x = ModRelu()(x)

        x = ComplexConv1d(128,5)(x)
        x = ModRelu()(x)

        x = ComplexAttention(128)(x)

        x = x.reshape(x.shape[0], -1)

        x = ComplexLinear(256)(x)
        x = ModRelu()(x)

        eigvals = ComplexLinear(self.K)(x).real

        psi = ComplexLinear(self.K*256)(x)
        psi = psi.reshape(-1,self.K,256)

        norm = jnp.abs(psi) ** 2
        norm = jnp.sqrt(norm.sum(axis=-1, keepdims=True)) + 1e-8
        psi = psi / norm
        
        return eigvals, psi
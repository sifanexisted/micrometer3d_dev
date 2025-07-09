import jax.numpy as jnp
from jax import random

import flax.linen as nn
from typing import Optional, Callable, Tuple
from .vit3d import MlpBlock


def normal(stddev=1e-2, dtype=jnp.float32) -> Callable:
    def init(key, shape, dtype=dtype):
        keys = random.split(key)
        return random.normal(keys[0], shape) * stddev

    return init


def complex_mul3d(x, weights_r, weights_i):
    """
    x: (b, h, w, d, c)
    weights_r: (c, out_dim, modes1, modes2, modes3)
    weights_i: (c, out_dim, modes1, modes2, modes3)
    out: (b, h, w, d, out_dim)
    """
    return jnp.einsum("bxyzc,coxyz->bxyzo", x, weights_r + 1j * weights_i)


# let's use this module to store weights
# essentially, we'll also use this when working with tucker tensorized weights
class FNO3DWeights(nn.Module):
    in_dim: int = 32
    out_dim: int = 32
    modes1: int = 12
    modes2: int = 12
    modes3: int = 12
    factorized: bool = False
    ranks: Optional[Tuple[float]] = None
    model_name: Optional[str] = None

    def setup(self):
        scale = 1 / (self.in_dim * self.out_dim)

        self.weights1_r = self.param(
            "weights1_r",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights1_i = self.param(
            "weights1_i",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights2_r = self.param(
            "weights2_r",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights2_i = self.param(
            "weights2_i",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )

        self.weights3_r = self.param(
            "weights3_r",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights3_i = self.param(
            "weights3_i",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights4_r = self.param(
            "weights4_r",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )
        self.weights4_i = self.param(
            "weights4_i",
            normal(scale, jnp.float32),
            (self.in_dim, self.out_dim, self.modes1, self.modes2, self.modes3),
            jnp.float32,
        )

    def __call__(self):
        return self.weights1_r, self.weights1_i, self.weights2_r, self.weights2_i, self.weights3_r, self.weights3_i, self.weights4_r, self.weights4_i


class SpectralConv3d(nn.Module):
    out_dim: int = 32
    modes1: int = 12
    modes2: int = 12
    modes3: int = 12

    @nn.compact
    def __call__(self, x):
        # x.shape: (b, h, w, d, c)

        # Initialize parameters
        in_dim = x.shape[-1]
        h = x.shape[1]
        w = x.shape[2]
        d = x.shape[3]

        # Checking that the modes are not more than the input size
        assert self.modes1 <= h // 2 + 1
        assert self.modes2 <= w // 2 + 1
        assert self.modes3 <= d // 2 + 1

        weights1_r, weights1_i, weights2_r, weights2_i, weights3_r, weights3_i, weights4_r, weights4_i = FNO3DWeights(
            in_dim=in_dim,
            out_dim=self.out_dim,
            modes1=self.modes1,
            modes2=self.modes2,
            modes3=self.modes3,
            factorized=False,  # TODO: create as input parameter
            ranks=(1, 1, 0.25, 0.25, 0.25),  # TODO: create as input parameter
            model_name='FNO3DWeights'
        )()

        # Compute Fourier coefficients
        x_ft = jnp.fft.rfftn(x, axes=(1, 2, 3))
        # Multiply relevant Fourier modes
        out_ft = jnp.zeros_like(x_ft)

        out_ft = out_ft.at[:, : self.modes1, : self.modes2, : self.modes3, :].set(
            complex_mul3d(
                x_ft[:, : self.modes1, : self.modes2, : self.modes3, :],
                weights1_r,
                weights1_i,
            )
        )
        out_ft = out_ft.at[:, -self.modes1:, : self.modes2, : self.modes3, :].set(
            complex_mul3d(
                x_ft[:, -self.modes1:, : self.modes2, : self.modes3, :],
                weights2_r,
                weights2_i,
            )
        )
        out_ft = out_ft.at[:, : self.modes1, -self.modes2:, : self.modes3, :].set(
            complex_mul3d(
                x_ft[:, : self.modes1, -self.modes2:, : self.modes3, :],
                weights3_r,
                weights3_i,
            )
        )
        out_ft = out_ft.at[:, -self.modes1:, -self.modes2:, : self.modes3, :].set(
            complex_mul3d(
                x_ft[:, -self.modes1:, -self.modes2:, : self.modes3, :],
                weights4_r,
                weights4_i,
            )
        )

        # Return to physical space
        x = jnp.fft.irfftn(out_ft, axes=(1, 2, 3), s=(h, w, d))
        return x


# trying out some changes from https://arxiv.org/pdf/2310.00120
class FourierStage(nn.Module):
    emb_dim: int = 32
    modes1: int = 12
    modes2: int = 12
    modes3: int = 12
    activation: Callable = nn.gelu
    factorized: bool = False

    @nn.compact
    def __call__(self, x):
        x_fourier = SpectralConv3d(
            out_dim=self.emb_dim,
            modes1=self.modes1,
            modes2=self.modes2,
            modes3=self.modes3,
        )(x)

        x_fourier = nn.LayerNorm()(x_fourier)
        x_local = nn.Conv(
            self.emb_dim,
            (1, 1, 1),
        )(x)

        x1 = self.activation(x_fourier + x_local)
        x2 = MlpBlock(self.emb_dim, self.emb_dim)(x1)
        x2 = nn.LayerNorm()(x2)

        x2_local = nn.Conv(
            self.emb_dim,
            (1, 1, 1),
        )(x)
        x3 = self.activation(x2 + x2_local)
        return x3


class FNO3d(nn.Module):
    modes1: int = 12
    modes2: int = 12
    modes3: int = 12
    emb_dim: int = 32
    out_dim: int = 1
    depth: int = 4
    activation: Callable = nn.gelu
    padding: int = 0  # Padding for non-periodic inputs
    model_name: Optional[str] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Lift the input to a higher dimension
        batch_grid = self.get_grid(x)
        x = jnp.concatenate([x, batch_grid], axis=-1)
        x = nn.Dense(self.emb_dim)(x)

        # Pad input
        if self.padding > 0:
            x = jnp.pad(
                x,
                (
                    (0, 0),
                    (0, self.padding),
                    (0, self.padding),
                    (0, self.padding),
                    (0, 0),
                ),
                mode="constant",
            )

        for _ in range(self.depth):
            x = FourierStage(
                emb_dim=self.emb_dim,
                modes1=self.modes1,
                modes2=self.modes2,
                modes3=self.modes3,
                activation=self.activation,
            )(x)

        # Unpad
        if self.padding > 0:
            x = x[:, : -self.padding, : -self.padding, : -self.padding, :]

        # Project to the output dimension
        x = nn.Dense(self.emb_dim)(x)
        x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)

        return x

    @staticmethod
    def get_grid(x):
        x1 = jnp.linspace(0, 1, x.shape[1])
        x2 = jnp.linspace(0, 1, x.shape[2])
        x3 = jnp.linspace(0, 1, x.shape[3])
        x1, x2 = jnp.meshgrid(x1, x2, x3, indexing='ij')
        grid = jnp.expand_dims(jnp.stack([x1, x2,  x3], axis=-1), 0)
        batched_grid = jnp.repeat(grid, x.shape[0], axis=0)
        return batched_grid

import os
from functools import partial

import h5py
import numpy as np
from einops import rearrange, repeat

import jax
import jax.numpy as jnp

from jax import random, jit

import torch
from torch.utils.data import Dataset, DataLoader, Subset


import h5py
import numpy as np
from torch.utils.data import Dataset

class PlainDataset(Dataset):
    def __init__(
        self,
        input_files,
        output_files,
        input_keys,
        output_keys,
        downsample_factor=2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        input_list = []
        output_list = []

        # Load and concatenate all input files
        for input_file, input_key in zip(input_files, input_keys):
            with h5py.File(input_file, "r") as f:
                input_list.append(np.array(f[input_key]))

        # Load and concatenate all output files
        for output_file, output_key in zip(output_files, output_keys):
            with h5py.File(output_file, "r") as f:
                output_list.append(np.array(f[output_key]))

        # Concatenate across the first axis (samples)
        self.inputs = np.concatenate(input_list, axis=0)
        self.outputs = np.concatenate(output_list, axis=0)

        # Sanity check
        assert len(self.inputs) == len(self.outputs), "Mismatched input/output lengths"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]



class BaseDataset(Dataset):
    # This dataset class is used for training VIT, FNO, UNet models.
    def __init__(
        self,
        input_files,
        output_files,
        input_keys,
        output_keys,
        downsample_factor=2
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        input_list = []
        output_list = []

        # Load and concatenate all input files
        for input_file, input_key in zip(input_files, input_keys):
            with h5py.File(input_file, "r") as f:
                input_list.append(np.array(f[input_key]))

        # Load and concatenate all output files
        for output_file, output_key in zip(output_files, output_keys):
            with h5py.File(output_file, "r") as f:
                output_list.append(np.array(f[output_key]))

        # Concatenate across the first axis (samples)
        self.inputs = np.vstack(input_list)
        self.outputs = np.vstack(output_list)

        print("Inputs shape:", self.inputs.shape)
        print("Outputs shape:", self.outputs.shape)



        b, c, d, h, w = self.outputs.shape
        self.h = h // self.downsample_factor
        self.w = w // self.downsample_factor
        self.d = d // self.downsample_factor

        x_star = np.linspace(0, 1, self.h)
        y_star = np.linspace(0, 1, self.w)
        z_star = np.linspace(0, 1, self.d)
        x_star, y_star, z_star = np.meshgrid(x_star, y_star, z_star, indexing="ij")

        self.grid = np.stack([x_star, y_star, z_star], axis=-1)
        self.coords = np.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None], z_star.flatten()[:, None]])

        self.inputs = np.expand_dims(self.inputs, axis=1)  # Add channel dimension
        self.inputs = np.transpose(self.inputs, (0, 3, 4, 2, 1))  # Convert to (H, W, C)
        self.outputs = np.transpose(self.outputs, (0, 3, 4, 2, 1)) # Co  nvert to (H, W, D, C)

        # Concatenate the grid and labels to the inputs
        self.inputs = self.inputs[
                       :, :: self.downsample_factor, :: self.downsample_factor, :: self.downsample_factor
                       ]
        self.outputs = self.outputs[
                        :, :: self.downsample_factor, :: self.downsample_factor, :: self.downsample_factor
                        ]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        # Get the original batch inputs, outputs, and labels
        batch_inputs, batch_outputs = self.inputs[index], self.outputs[index]

        return batch_inputs, batch_outputs


class CViTDataset(BaseDataset):
    def __init__(
        self,
        input_files,
        output_files,
        input_keys,
        output_keys,
        downsample_factor=2,
        num_query_points=None,
        channel_last=True,
    ):
        super().__init__(
            input_files,
            output_files,
            input_keys,
            output_keys,
            downsample_factor,
            channel_last,
        )
        self.num_query_points = num_query_points

    def __getitem__(self, index):
        batch_inputs, batch_outputs = super().__getitem__(index)
        batch_outputs = rearrange(batch_outputs, "h w d c -> (h w d) c")

        if self.num_query_points is not None:
            query_index = np.random.choice(
                batch_outputs.shape[0], self.num_query_points, replace=False
            )
            batch_coords = self.coords[query_index]
            batch_outputs = batch_outputs[query_index]
        else:
            batch_coords = self.coords

        return batch_coords, batch_inputs, batch_outputs


def generate_paths_and_keys(data_path, split, suffixes):
    inputs_path = os.path.join(data_path, f"{split}_inputs")
    outputs_path = os.path.join(data_path, f"{split}_outputs")

    input_keys = [f"cmme_3D_{split}_inputs_{suffix}" for suffix in suffixes]
    output_keys = [f"cmme_3D_{split}_outputs_{suffix}" for suffix in suffixes]

    input_files = [os.path.join(inputs_path, f"{key}.mat") for key in input_keys]
    output_files = [os.path.join(outputs_path, f"{key}.mat") for key in output_keys]

    return {
        "input_files": input_files,
        "output_files": output_files,
        "input_keys": input_keys,
        "output_keys": output_keys,
    }


def create_datasets(config):
    data_path = config.dataset.data_path

    train_suffixes = config.dataset.train_files
    test_suffixes = config.dataset.test_files

    train_data = generate_paths_and_keys(data_path, "train", train_suffixes)
    test_data = generate_paths_and_keys(data_path, "test", test_suffixes)

    train_dataset = BaseDataset(
        train_data["input_files"],
        train_data["output_files"],
        train_data["input_keys"],
        train_data["output_keys"],
        downsample_factor=config.dataset.downsample_factor,
    )

    test_dataset = BaseDataset(
        test_data["input_files"],
        test_data["output_files"],
        test_data["input_keys"],
        test_data["output_keys"],
        downsample_factor=config.dataset.downsample_factor,
    )

    if config.dataset.train_samples < len(train_dataset):
        train_indices = torch.randperm(len(train_dataset))[
            : config.dataset.train_samples
        ]
        train_dataset = Subset(train_dataset, train_indices)

    if config.dataset.test_samples < len(test_dataset):
        test_indices = torch.randperm(len(test_dataset))[: config.dataset.test_samples]
        test_dataset = Subset(test_dataset, test_indices)

    return train_dataset, test_dataset


def create_dataloaders(config, train_dataset, test_dataset):
    num_devices = jax.device_count()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dataset.train_batch_size * num_devices,
        num_workers=config.dataset.num_workers,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.dataset.test_batch_size * num_devices,
        num_workers=config.dataset.num_workers,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, test_loader


class BatchParser:
    def __init__(self, config, batch):
        self.config = config
        self.num_query_points = config.training.num_query_points

        b, h, w, d, c = batch[1].shape
        x_star = jnp.linspace(0, 1, h)
        y_star = jnp.linspace(0, 1, w)
        z_star = jnp.linspace(0, 1, d)
        x_star, y_star, z_star = jnp.meshgrid(x_star, y_star, z_star, indexing="ij")

        self.coords = jnp.hstack([x_star.flatten()[:, None], y_star.flatten()[:, None], z_star.flatten()[:, None]])

    @partial(jit, static_argnums=(0,))
    def random_query(self, batch, rng_key=None):
        batch_inputs, batch_outputs = batch
        batch_outputs = rearrange(batch_outputs, "b h w d c -> b (h w d) c")

        query_index = random.choice(
            rng_key, batch_outputs.shape[1], (self.num_query_points,), replace=False
        )
        batch_coords = self.coords[query_index]
        batch_outputs = batch_outputs[:, query_index]

        # Repeat the coords  across devices
        batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

        return batch_coords, batch_inputs, batch_outputs

    @partial(jit, static_argnums=(0,))
    def query_all(self, batch):
        batch_inputs, batch_outputs = batch

        if self.config.model.model_name.lower().startswith("cvit"):
            batch_outputs = rearrange(batch_outputs, "b h w d c -> b (h w d) c")
            batch_coords = self.coords

            # Repeat the coords  across devices
            batch_coords = repeat(batch_coords, "b d -> n b d", n=jax.device_count())

            return batch_coords, batch_inputs, batch_outputs

        return batch_inputs, batch_outputs

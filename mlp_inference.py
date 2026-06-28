"""Pure-numpy inference for Keras Sequential MLP models (.keras archives)."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass
class DenseLayer:
    weights: np.ndarray
    bias: np.ndarray
    activation: str


class KerasNumpyMLP:
    """Run forward passes without importing TensorFlow."""

    def __init__(self, layers: list[DenseLayer]):
        self.layers = layers

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=np.float32)
        for layer in self.layers:
            out = out @ layer.weights + layer.bias
            if layer.activation == "relu":
                out = np.maximum(out, 0.0)
        return out


def load_keras_mlp(path: Path) -> KerasNumpyMLP:
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
        dense_layers = [
            layer
            for layer in config["config"]["layers"]
            if layer["class_name"] == "Dense"
        ]
        activations = [layer["config"]["activation"] for layer in dense_layers]

        with archive.open("model.weights.h5") as weights_file:
            with h5py.File(weights_file, "r") as weights:
                keras_layers = [
                    DenseLayer(
                        weights=weights[f"layers/dense{suffix}/vars/0"][:],
                        bias=weights[f"layers/dense{suffix}/vars/1"][:],
                        activation=activation,
                    )
                    for suffix, activation in zip(_dense_suffixes(len(dense_layers)), activations)
                ]

    return KerasNumpyMLP(keras_layers)


def _dense_suffixes(count: int) -> list[str]:
    if count == 0:
        return []
    return [""] + [f"_{i}" for i in range(1, count)]

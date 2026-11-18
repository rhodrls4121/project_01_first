"""Feed-forward individuals with one weight and bias vector per neuron."""

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, tanh
from random import Random


@dataclass(frozen=True)
class Individual:
    layers: tuple[int, ...]
    genes: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.layers) < 2 or any(type(n) is not int or n < 1 for n in self.layers):
            raise ValueError("layers must contain at least two positive integer widths")
        if len(self.genes) != self.gene_count(self.layers):
            raise ValueError("gene count does not match the layer widths")
        if any(type(gene) not in (int, float) or not isfinite(gene) for gene in self.genes):
            raise ValueError("genes must be finite numbers")

    @staticmethod
    def gene_count(layers: tuple[int, ...]) -> int:
        return sum((inputs + 1) * outputs for inputs, outputs in pairwise(layers))

    @classmethod
    def random(cls, layers: tuple[int, ...], rng: Random) -> "Individual":
        return cls(layers, tuple(rng.uniform(-1, 1) for _ in range(cls.gene_count(layers))))

    def __call__(self, inputs: Iterable[float]) -> tuple[float, ...]:
        values = tuple(map(float, inputs))
        if len(values) != self.layers[0] or not all(map(isfinite, values)):
            raise ValueError(f"expected {self.layers[0]} finite inputs")
        genes = iter(self.genes)
        for layer, width in enumerate(self.layers[1:], start=1):
            outputs = []
            for _ in range(width):
                weighted_sum = sum(next(genes) * value for value in values) + next(genes)
                outputs.append(weighted_sum if layer == len(self.layers) - 1 else tanh(weighted_sum))
            values = tuple(outputs)
        if not all(map(isfinite, values)):
            raise ValueError("network produced a non-finite output")
        return values

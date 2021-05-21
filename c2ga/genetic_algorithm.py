from typing import Callable

import numpy as np


class GeneticAlgorithm:
    crossover_rate: float
    mutation_rate: float
    gene_count: int

    generator: Callable
    objective_fn: Callable

    def __init__(self, cr: float, mr: float, count: int, generator: Callable, fn: Callable):
        self.crossover_rate = cr
        self.mutation_rate = mr
        self.gene_count = count
        self.generator = generator
        self.objective_fn = fn

    @staticmethod
    def fitness():
        # TODO
        return

    @staticmethod
    def generate_individual(size: int, low: float = 0.0, high: float = 1.0):
        return np.random.uniform(low, high, size)

    @staticmethod
    def generate_population(count: int, size: int, low: float = 0.0, high: float = 1.0):
        return np.array([
            GeneticAlgorithm.generate_individual(size, low, high) for _ in range(count)
        ])

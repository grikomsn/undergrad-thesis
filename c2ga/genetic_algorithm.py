from typing import Callable


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

"""XOR training settings and fitness function."""

from neuroevolution import Config, Individual, configure


XOR_CASES = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))


def score_xor(individual: Individual) -> float:
    errors = [(individual(inputs)[0] - target) ** 2 for inputs, target in XOR_CASES]
    return -sum(errors) / len(errors)


def settings(seed: int, debug: bool) -> Config:
    return configure(
        input_size=2, hidden_layers=(4,), output_size=1,
        generations=160, population_size=100, parent_count=20, survivor_count=5,
        crossover="uniform", mutation_rate=0.2, mutation_range=(-0.8, 0.8),
        seed=seed, debug=debug,
    )

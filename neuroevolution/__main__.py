"""Edit this example to define your own network and scoring function."""

import argparse

from . import Individual, configure, evolve, load_individual, run_saved


XOR_CASES = (((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0))


def score_xor(individual: Individual) -> float:
    errors = [(individual(inputs)[0] - target) ** 2 for inputs, target in XOR_CASES]
    return -sum(errors) / len(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the XOR example or a saved individual.")
    parser.add_argument("--model", default="best_individual.json", help="JSON model path")
    parser.add_argument("--predict", nargs="+", type=float, help="input values for a saved model")
    parser.add_argument("--quiet", action="store_true", help="hide generation progress")
    args = parser.parse_args()
    if args.predict is not None:
        print(run_saved(args.model, args.predict))
        return

    config = configure(
        input_size=2, hidden_layers=(4,), output_size=1,
        generations=160, population_size=100, parent_count=20, survivor_count=5,
        crossover="uniform", mutation_rate=0.2, mutation_range=(-0.8, 0.8),
        seed=42, debug=not args.quiet,
    )
    result = evolve(config, score_xor, save_path=args.model)
    print(f"Best score: {result.score:.6f} (generation {result.generation})")
    print(f"Saved to {args.model}")
    individual = load_individual(args.model)
    for inputs, target in XOR_CASES:
        output, = individual(inputs)
        print(f"{inputs} -> {output:.6f} (expected {target})")


if __name__ == "__main__":
    main()

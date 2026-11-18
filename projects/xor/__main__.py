"""Edit this example to define your own network and scoring function."""

import argparse
from pathlib import Path

from neuroevolution import evolve, load_individual, run_saved

from .training import XOR_CASES, score_xor, settings


MODEL_DIR = Path(__file__).resolve().parent / "models"


def main(default_model: str | Path = MODEL_DIR / "champion.json") -> None:
    parser = argparse.ArgumentParser(description="Run the XOR example or a saved individual.")
    parser.add_argument("--model", default=default_model, help="JSON model path")
    parser.add_argument("--predict", nargs="+", type=float, help="input values for a saved model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="hide generation progress")
    args = parser.parse_args()
    if args.predict is not None:
        try:
            print(run_saved(args.model, args.predict))
        except FileNotFoundError:
            parser.error(f"model file not found: {args.model}. Run training first or pass --model.")
        return

    config = settings(seed=args.seed, debug=not args.quiet)
    result = evolve(config, score_xor, save_path=args.model)
    print(f"Best score: {result.score:.6f} (generation {result.generation})")
    print(f"Saved to {args.model}")
    individual = load_individual(args.model)
    for inputs, target in XOR_CASES:
        output, = individual(inputs)
        print(f"{inputs} -> {output:.6f} (expected {target})")


if __name__ == "__main__":
    main()

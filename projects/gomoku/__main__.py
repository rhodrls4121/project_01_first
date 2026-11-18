"""Train an AI by self-play or play against a saved model."""

import argparse
from pathlib import Path

from neuroevolution import evolve

from .rules import Rules
from .training import Tournament, settings


MODEL_DIR = Path(__file__).resolve().parent / "models"


def default_model(rules: Rules) -> Path:
    return MODEL_DIR / f"{rules.value}-champion.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="오목 개체를 학습시키거나 저장 개체와 대국합니다.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="개체끼리 대국해 학습")
    train.add_argument("--rules", choices=[rule.value for rule in Rules], default=Rules.RENJU.value)
    train.add_argument("--generations", type=int, default=12)
    train.add_argument("--population", type=int, default=12)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--model", type=Path)
    train.add_argument("--quiet", action="store_true")
    play = commands.add_parser("play", help="브라우저에서 저장 개체와 대국")
    play.add_argument("--port", type=int, default=8000)
    play.add_argument("--model", type=Path, help="두 규칙에 사용할 사용자 모델")
    args = parser.parse_args()

    try:
        if args.command == "play":
            from .web import serve

            models = {rule: args.model or default_model(rule) for rule in Rules}
            serve(args.port, models)
            return

        rules = Rules(args.rules)
        config = settings(args.generations, args.population, args.seed, not args.quiet)
        path = args.model or default_model(rules)
        result = evolve(config, population_score=Tournament(rules, args.seed), save_path=path)
        print(f"최종 {result.generation}세대 1위 | 경기당 점수 {result.score:.3f}")
        print(f"모델 저장: {path}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

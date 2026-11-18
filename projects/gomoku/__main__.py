"""Train an AI by self-play or play against a saved model."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from neuroevolution import evolve, load_checkpoint

from .rules import Rules
from .training import Tournament, settings


MODEL_DIR = Path(__file__).resolve().parent / "models"
RUN_IDS = {rule: f"gomoku:{rule.value}:v1" for rule in Rules}


def default_model(rules: Rules) -> Path:
    return MODEL_DIR / f"{rules.value}-champion.json"


def train_model(args: argparse.Namespace) -> None:
    if args.generations < 1:
        raise ValueError("세대 수는 1 이상이어야 합니다.")
    resume = load_checkpoint(args.resume) if args.resume is not None else None
    if resume is None:
        rules = Rules(args.rules or Rules.RENJU)
        seed = 42 if args.seed is None else args.seed
        population = 12 if args.population is None else args.population
        config = settings(args.generations, population, seed, not args.quiet)
        completed = 0
    else:
        rules = next((rule for rule, run_id in RUN_IDS.items() if run_id == resume.run_id), None)
        if rules is None or not resume.competitive or resume.config.seed is None:
            raise ValueError("오목 자가 대국 체크포인트를 선택하세요.")
        if args.rules is not None and args.rules != rules.value:
            raise ValueError("체크포인트와 대국 규칙이 다릅니다.")
        seed = resume.config.seed
        if args.seed is not None and args.seed != seed:
            raise ValueError("체크포인트와 시드가 다릅니다.")
        if args.population is not None and args.population != resume.config.population_size:
            raise ValueError("체크포인트와 개체 수가 다릅니다.")
        completed = resume.generation
        config = replace(resume.config, generations=completed + args.generations, debug=not args.quiet)

    path = args.model or default_model(rules)
    checkpoint_path = args.checkpoint or args.resume or Path(f"gomoku-{rules.value}-checkpoint.json")
    if args.resume is not None and args.resume.resolve() == path.resolve():
        raise ValueError("모델과 재개용 체크포인트의 경로를 다르게 지정하세요.")
    print(f"{rules.value} | {' → '.join(map(str, config.layers))} | "
          f"{completed + 1}~{config.generations}세대 | 체크포인트: {checkpoint_path}", flush=True)
    try:
        result = evolve(
            config, population_score=Tournament(rules, seed, completed),
            save_path=path, checkpoint_path=checkpoint_path, resume=resume, run_id=RUN_IDS[rules],
        )
    except KeyboardInterrupt:
        print("\n학습을 중단했습니다. 완료된 세대가 있다면 체크포인트에서 --resume으로 재개할 수 있습니다.",
              file=sys.stderr)
        raise SystemExit(130) from None
    print(f"최종 {result.generation}세대 1위 | 경기당 점수 {result.score:.3f}")
    print(f"모델 저장: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="오목 개체를 학습시키거나 저장 개체와 대국합니다.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="개체끼리 대국해 학습")
    train.add_argument("--rules", choices=[rule.value for rule in Rules])
    train.add_argument("--generations", type=int, default=12, help="새로 진행할 세대 수")
    train.add_argument("--population", type=int)
    train.add_argument("--seed", type=int)
    train.add_argument("--resume", type=Path, help="완료된 세대의 체크포인트")
    train.add_argument("--checkpoint", type=Path, help="체크포인트 저장 경로")
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

        train_model(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

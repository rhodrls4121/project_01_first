"""Balanced, generation-local round-robin self-play."""

from itertools import combinations
from random import Random

from neuroevolution import Config, Individual, configure
from neuroevolution.evolution import ReportProgress

from .policy import FEATURES, choose_move
from .rules import BLACK, CENTER, EMPTY, SIZE, WHITE, Board, Rules


def settings(generations: int, population_size: int, seed: int, debug: bool) -> Config:
    return configure(
        input_size=FEATURES, hidden_layers=(24, 12), output_size=1,
        generations=generations, population_size=population_size,
        parent_count=max(2, population_size // 3),
        survivor_count=max(1, population_size // 6),
        crossover="uniform", mutation_rate=0.15, mutation_range=(-0.5, 0.5),
        seed=seed, debug=debug,
    )


def opening(rng: Random) -> tuple[int, ...]:
    """Sample a small central opening, shared by the two color assignments."""
    white = CENTER + rng.choice((1, SIZE + 1))
    black = CENTER + rng.choice(
        tuple(row * SIZE + col for row in range(-2, 3) for col in range(-2, 3)
              if row * SIZE + col not in (0, white - CENTER))
    )
    return CENTER, white, black


def duel(black: Individual, white: Individual, rules: Rules, moves: tuple[int, ...]) -> int:
    board = Board(rules)
    for point in moves:
        board.play(point)
    while not board.finished:
        player = black if board.turn == BLACK else white
        board.play(choose_move(board, player))
    return board.winner


class Tournament:
    def __init__(self, rules: Rules, seed: int, completed: int = 0) -> None:
        self.rules = rules
        self.seed = seed
        self.completed = completed

    def __call__(self, population: tuple[Individual, ...], report: ReportProgress) -> list[float]:
        rng = Random(f"{self.seed}:{self.completed + 1}")
        count = len(population)
        total = count * (count - 1)
        scores = [0.0] * count
        done = 0
        report(done, total)
        for first, second in combinations(range(count), 2):
            moves = opening(rng)
            for black, white in ((first, second), (second, first)):
                winner = duel(population[black], population[white], self.rules, moves)
                scores[black] += 1 if winner == BLACK else 0.5 if winner == EMPTY else 0
                scores[white] += 1 if winner == WHITE else 0.5 if winner == EMPTY else 0
                done += 1
                report(done, total)
        self.completed += 1
        return [points / (2 * (count - 1)) for points in scores]

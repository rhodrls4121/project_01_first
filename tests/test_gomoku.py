import unittest
from unittest.mock import patch

from neuroevolution import Individual, configure, evolve
from projects.gomoku.rules import BLACK, CENTER, EMPTY, SIZE, WHITE, Board, Rules
from projects.gomoku.training import Tournament


def at(row: int, col: int) -> int:
    return row * SIZE + col


class RulesTest(unittest.TestCase):
    def position(self, black=(), white=(), rules=Rules.RENJU):
        board = Board(rules)
        board.play(CENTER)
        board.cells[CENTER] = EMPTY
        for row, col in black:
            board.cells[at(row, col)] = BLACK
        for row, col in white:
            board.cells[at(row, col)] = WHITE
        board.turn = BLACK
        return board

    def test_renju_first_move_is_central(self):
        board = Board(Rules.RENJU)
        with self.assertRaises(ValueError):
            board.play(CENTER - 1)
        board.play(CENTER)
        with self.assertRaises(ValueError):
            board.play(CENTER)
        self.assertEqual(board.turn, WHITE)

    def test_overline_depends_on_rules_and_color(self):
        run = [(7, col) for col in (4, 5, 6, 8, 9)]
        for rules, color, winner, foul in (
            (Rules.RENJU, BLACK, WHITE, "장목"),
            (Rules.RENJU, WHITE, WHITE, None),
            (Rules.FREESTYLE, BLACK, BLACK, None),
        ):
            with self.subTest(rules=rules, color=color):
                board = self.position(
                    black=run if color == BLACK else (),
                    white=run if color == WHITE else (),
                    rules=rules,
                )
                board.turn = color
                board.play(CENTER)
                self.assertEqual((board.winner, board.foul), (winner, foul))

    def test_exact_five_takes_precedence_over_overline(self):
        horizontal = [(7, col) for col in (4, 5, 6, 8, 9)]
        vertical = [(row, 7) for row in (5, 6, 8, 9)]
        board = self.position(black=horizontal + vertical)
        board.play(CENTER)
        self.assertEqual((board.winner, board.foul), (BLACK, None))

    def test_one_straight_four_is_legal(self):
        board = self.position(black=[(7, 5), (7, 6), (7, 8)])
        self.assertIsNone(board.forbidden(CENTER))
        self.assertEqual(board.cells[CENTER], EMPTY)

    def test_two_distinct_fours_can_share_an_axis(self):
        board = self.position(black=[(7, col) for col in (4, 6, 8, 10)])
        self.assertEqual(board.forbidden(CENTER), "4-4")

    def test_double_three_and_blocked_extension(self):
        cross = [(7, 6), (7, 8), (6, 7), (8, 7)]
        board = self.position(black=cross)
        self.assertEqual(board.forbidden(CENTER), "3-3")
        board = self.position(black=cross, white=[(7, 4), (7, 10)])
        self.assertIsNone(board.forbidden(CENTER))

    def test_three_extension_with_overline_is_forbidden(self):
        cross = [(7, 6), (7, 8), (6, 7), (8, 7)]
        blocked_extension = [(row, 5) for row in (4, 5, 6, 8, 9)]
        board = self.position(black=cross + blocked_extension, white=[(7, 10)])
        self.assertIsNone(board.forbidden(CENTER))


class TournamentTest(unittest.TestCase):
    def test_same_opening_and_both_colors_with_normalized_points(self):
        first = Individual((1, 1), (1, 0))
        second = Individual((1, 1), (2, 0))
        progress = []
        with patch("projects.gomoku.training.duel", side_effect=[BLACK, EMPTY]) as duel:
            scores = Tournament(Rules.RENJU, 5)(
                (first, second), lambda done, total: progress.append((done, total)),
            )
        game_one, game_two = (call.args for call in duel.call_args_list)
        self.assertEqual(game_one, (first, second, Rules.RENJU, game_two[3]))
        self.assertEqual(game_two[:2], (second, first))
        self.assertEqual(scores, [0.75, 0.25])
        self.assertEqual(progress, [(0, 2), (1, 2), (2, 2)])

    def test_population_evaluation_saves_last_generation_leader(self):
        config = configure(input_size=1, hidden_layers=(), output_size=1,
                           generations=2, population_size=2, parent_count=2,
                           survivor_count=1, seed=7, debug=False)
        scores = iter(((10, 0), (0, 1)))
        populations = []

        def evaluate(population, report):
            populations.append(population)
            return next(scores)

        result = evolve(config, population_score=evaluate, save_path=None)
        self.assertEqual(result.generation, 2)
        self.assertIs(result.individual, populations[-1][1])
        self.assertEqual(result.score, 1)

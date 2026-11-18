from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from neuroevolution import configure, evolve, load_checkpoint
from projects.gomoku.rules import Rules
from projects.gomoku.training import Tournament, settings


class CheckpointTest(unittest.TestCase):
    def test_interruption_keeps_last_complete_generation(self):
        config = configure(input_size=1, hidden_layers=(2,), output_size=1,
                           generations=4, population_size=4, parent_count=2,
                           survivor_count=1, seed=7, debug=False)

        def score(individual):
            return -abs(individual((1,))[0])

        calls = 0

        def interrupt(individual):
            nonlocal calls
            calls += 1
            if calls == config.population_size + 2:
                raise KeyboardInterrupt
            return score(individual)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            with self.assertRaises(KeyboardInterrupt):
                evolve(config, interrupt, save_path=None, checkpoint_path=path)
            snapshot = load_checkpoint(path)
            self.assertEqual(snapshot.generation, 1)
            split = evolve(config, score, save_path=None, checkpoint_path=path, resume=snapshot)
            self.assertEqual(load_checkpoint(path).generation, 4)
            full = evolve(config, score, save_path=None)
            self.assertEqual(full, split)

    def test_competitive_resume_replays_same_openings_and_final_population(self):
        seed = 14
        rules = Rules.RENJU
        config = settings(3, 4, seed, False)
        with TemporaryDirectory() as directory:
            full_path = Path(directory) / "full.json"
            split_path = Path(directory) / "split.json"
            full = evolve(config, population_score=Tournament(rules, seed),
                          save_path=None, checkpoint_path=full_path, run_id="renju")
            evolve(replace(config, generations=1), population_score=Tournament(rules, seed),
                   save_path=None, checkpoint_path=split_path, run_id="renju")
            snapshot = load_checkpoint(split_path)
            split = evolve(config, population_score=Tournament(rules, seed, snapshot.generation),
                           save_path=None, checkpoint_path=split_path, resume=snapshot, run_id="renju")
            self.assertEqual(full, split)
            self.assertEqual(load_checkpoint(full_path), load_checkpoint(split_path))

            with self.assertRaisesRegex(ValueError, "run_id"):
                evolve(config, population_score=Tournament(rules, seed, snapshot.generation),
                       save_path=None, resume=snapshot, run_id="freestyle")

    def test_checkpoint_cannot_overwrite_model(self):
        config = settings(1, 2, 42, False)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "same.json"
            with self.assertRaisesRegex(ValueError, "paths must be different"):
                evolve(config, lambda _: 0, save_path=path, checkpoint_path=path)
            self.assertFalse(path.exists())

"""Display transient evaluation progress and persistent generation summaries."""

from collections.abc import Sequence
from math import fsum
import sys


class Progress:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.width = 0

    def update(self, generation: int, generations: int, done: int, total: int) -> None:
        if self.enabled:
            line = f"Generation {generation}/{generations} | {done / total:6.1%} ({done}/{total})"
            self.width = max(self.width, len(line))
            print("\r" + line.ljust(self.width), end="", file=sys.stderr, flush=True)

    def finish(self, generation: int, generations: int, scores: Sequence[float], best: float) -> None:
        if self.enabled:
            mean = fsum(score / len(scores) for score in scores)
            line = (
                f"Generation {generation}/{generations} | 100% | "
                f"best={max(scores):.6g} mean={mean:.6g} "
                f"worst={min(scores):.6g} overall_best={best:.6g}"
            )
            print("\r" + line.ljust(self.width), file=sys.stderr, flush=True)
            self.width = 0

    def clear(self) -> None:
        if self.width:
            print("\r" + " " * self.width + "\r", end="", file=sys.stderr, flush=True)
            self.width = 0

"""15×15 freestyle Gomoku and Renju placement rules."""

from enum import Enum

SIZE = 15
AREA = SIZE * SIZE
CENTER = AREA // 2
EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class Rules(str, Enum):
    FREESTYLE = "freestyle"
    RENJU = "renju"


def _ray(point: int, dr: int, dc: int) -> tuple[int, ...]:
    row, col = divmod(point, SIZE)
    points = []
    for _ in range(5):
        row, col = row + dr, col + dc
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            break
        points.append(row * SIZE + col)
    return tuple(points)


# Five points per side suffice to distinguish five from an overline.
RAYS = tuple(
    tuple((_ray(point, -dr, -dc), _ray(point, dr, dc)) for dr, dc in DIRECTIONS)
    for point in range(AREA)
)
NEIGHBORS = tuple(
    tuple(ray[0] for axis in rays for ray in axis if ray)
    for rays in RAYS
)
LINES = tuple(
    tuple((left[::-1] + (point,) + right, len(left)) for left, right in rays)
    for point, rays in enumerate(RAYS)
)


def opponent(color: int) -> int:
    return BLACK + WHITE - color


def winning_shape(shapes: tuple[tuple[int, int], ...], rules: Rules, color: int) -> bool:
    if rules is Rules.RENJU and color == BLACK:
        return any(length == 5 for length, _ in shapes)
    return any(length >= 5 for length, _ in shapes)


class Board:
    def __init__(self, rules: Rules) -> None:
        self.rules = Rules(rules)
        self.cells = [EMPTY] * AREA
        self.moves: list[int] = []
        self.frontier: set[int] = set()
        self.turn = BLACK
        self.winner = EMPTY
        self.foul: str | None = None

    @property
    def finished(self) -> bool:
        return self.winner != EMPTY or len(self.moves) == AREA

    def shape(self, point: int, color: int) -> tuple[tuple[int, int], ...]:
        """Report run lengths and open ends for a stone imagined at point."""
        result = []
        for left, right in RAYS[point]:
            length, open_ends = 1, 0
            for side in (left, right):
                for index in side:
                    stone = self.cells[index]
                    if stone != color:
                        open_ends += stone == EMPTY
                        break
                    length += 1
            result.append((length, open_ends))
        return tuple(result)

    def _wins(self, point: int, color: int) -> bool:
        return winning_shape(self.shape(point, color), self.rules, color)

    def forbidden(self, point: int) -> str | None:
        """Classify a proposed black move on an empty intersection."""
        if self.rules is not Rules.RENJU:
            return None
        self.cells[point] = BLACK
        try:
            return self._black_foul(point)
        finally:
            self.cells[point] = EMPTY

    def _black_foul(self, point: int) -> str | None:
        lengths = [length for length, _ in self.shape(point, BLACK)]
        if 5 in lengths:
            return None
        if max(lengths) > 5:
            return "장목"

        fours = set()
        threes: dict[frozenset[int], set[int]] = {}
        for line, center in LINES[point]:
            # An open four has two winning windows but one set of four stones.
            for start in range(max(0, center - 4), min(center + 1, len(line) - 4)):
                window = line[start:start + 5]
                stones = frozenset(i for i in window if self.cells[i] == BLACK)
                if len(stones) != 4 or any(self.cells[i] == WHITE for i in window):
                    continue
                if start > 0 and self.cells[line[start - 1]] == BLACK:
                    continue
                if start + 5 < len(line) and self.cells[line[start + 5]] == BLACK:
                    continue
                fours.add(stones)
                if len(fours) > 1:
                    return "4-4"

            # A three must be extendable to four consecutive stones with two
            # exact-five completions. The extension itself must also be legal.
            for start in range(max(1, center - 3), min(center + 1, len(line) - 4)):
                ends = (line[start - 1], line[start + 4])
                if any(self.cells[i] != EMPTY for i in ends):
                    continue
                if start >= 2 and self.cells[line[start - 2]] == BLACK:
                    continue
                if start + 5 < len(line) and self.cells[line[start + 5]] == BLACK:
                    continue
                window = line[start:start + 4]
                stones = frozenset(i for i in window if self.cells[i] == BLACK)
                empty = [i for i in window if self.cells[i] == EMPTY]
                if len(stones) == 3 and len(empty) == 1:
                    threes.setdefault(stones, set()).add(empty[0])

        if len(threes) < 2:
            return None
        genuine = 0
        for extensions in threes.values():
            for extension in extensions:
                if not self._wins(extension, BLACK) and self.forbidden(extension) is None:
                    genuine += 1
                    if genuine == 2:
                        return "3-3"
                    break
        return None

    def play(self, point: int) -> None:
        if self.finished:
            raise ValueError("이미 끝난 대국입니다.")
        if type(point) is not int or not 0 <= point < AREA:
            raise ValueError("판 안의 교차점을 선택하세요.")
        if self.cells[point] != EMPTY:
            raise ValueError("이미 돌이 놓인 자리입니다.")
        if self.rules is Rules.RENJU and not self.moves and point != CENTER:
            raise ValueError("렌주 첫 수는 중앙에 놓아야 합니다.")
        color = self.turn
        self.foul = self.forbidden(point) if color == BLACK else None
        self.cells[point] = color
        self.moves.append(point)
        self.frontier.discard(point)
        self.frontier.update(i for i in NEIGHBORS[point] if self.cells[i] == EMPTY)
        if self.foul:
            self.winner = WHITE
        elif self._wins(point, color):
            self.winner = color
        self.turn = opponent(color)

"""Use a small neural network to rank candidate intersections."""

from neuroevolution import Individual

from .rules import AREA, BLACK, CENTER, EMPTY, SIZE, Board, opponent, winning_shape

FEATURES = 18
CENTER_DISTANCE = tuple(
    (abs(row - SIZE // 2) + abs(col - SIZE // 2)) / (SIZE - 1)
    for row in range(SIZE) for col in range(SIZE)
)


def validate_model(individual: Individual) -> None:
    if individual.layers[0] != FEATURES or individual.layers[-1] != 1:
        raise ValueError(f"오목 모델에는 입력 {FEATURES}개와 출력 1개가 필요합니다.")


def _features(
    own: tuple[tuple[int, int], ...], other: tuple[tuple[int, int], ...], point: int, moves: int
) -> tuple[float, ...]:
    lines = tuple(
        feature
        for side in (own, other)
        for length, ends in sorted(side, reverse=True)
        for feature in (min(length, 6) / 6, ends / 2)
    )
    return (*lines, CENTER_DISTANCE[point], moves / AREA)


def choose_move(board: Board, individual: Individual) -> int:
    if not board.moves:
        return CENTER

    color = board.turn
    other = opponent(color)
    ranked = []
    for point in sorted(board.frontier):
        own = board.shape(point, color)
        enemy = board.shape(point, other)
        if winning_shape(own, board.rules, color):
            return point
        threat = winning_shape(enemy, board.rules, other)
        score, = individual(_features(own, enemy, point, len(board.moves)))
        ranked.append((threat, score, -point))

    for _, _, negative_point in sorted(ranked, reverse=True):
        point = -negative_point
        if color == BLACK and board.forbidden(point):
            continue
        return point

    for point, stone in enumerate(board.cells):
        if stone == EMPTY and (color != BLACK or not board.forbidden(point)):
            return point
    # If every vacant point is forbidden, playing any of them loses by foul.
    return next(point for point, stone in enumerate(board.cells) if stone == EMPTY)

"""Serve the board and run the same policy and rules used during training."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path

from neuroevolution import Individual, load_individual

from .policy import choose_move, validate_model
from .rules import AREA, BLACK, WHITE, Board, Rules


def play_turn(payload: dict, models: dict[Rules, Individual]) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("잘못된 대국 요청입니다.")
    rules = Rules(payload.get("rules"))
    human = payload.get("human")
    moves = payload.get("moves")
    if type(human) is not int or human not in (BLACK, WHITE):
        raise ValueError("흑 또는 백을 선택하세요.")
    if not isinstance(moves, list) or len(moves) > AREA:
        raise ValueError("올바르지 않은 기보입니다.")
    board = Board(rules)
    for point in moves:
        board.play(point)
    if "move" in payload:
        if board.turn != human:
            raise ValueError("지금은 상대의 차례입니다.")
        board.play(payload["move"])
    elif moves:
        raise ValueError("새로 시작할 때는 빈 기보를 사용하세요.")
    if not board.finished and board.turn != human:
        board.play(choose_move(board, models[rules]))
    return {
        "cells": board.cells,
        "moves": board.moves,
        "finished": board.finished,
        "winner": board.winner,
        "foul": board.foul,
    }


def serve(port: int, paths: dict[Rules, Path]) -> None:
    if not 1 <= port <= 65535:
        raise ValueError("포트 번호는 1~65535 사이여야 합니다.")
    models = {rules: load_individual(path) for rules, path in paths.items()}
    for model in models.values():
        validate_model(model)
    page = Path(__file__).with_name("board.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(404)
                return
            self.reply(200, "text/html", page)

        def do_POST(self) -> None:
            if self.path != "/api/turn":
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/json":
                self.send_error(415)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("잘못된 요청 크기입니다.")
                payload = json.loads(self.rfile.read(size))
                response = play_turn(payload, models)
            except (ValueError, RecursionError) as error:
                self.reply(400, "application/json", json.dumps({"error": str(error)}).encode())
                return
            self.reply(200, "application/json", json.dumps(response).encode())

    with ThreadingHTTPServer(("0.0.0.0", port), Handler) as server:
        print(f"브라우저에서 http://localhost:{port} 열기", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n대국 서버를 종료했습니다.")

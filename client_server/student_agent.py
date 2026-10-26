"""
Student Agent for the Stones and Rivers game.

The game engine is supplied by the course.  This file contains only the
student-controlled decision making code.  The agent uses the same move rules
as the engine, then combines tactical checks, a positional evaluation, and a
small time-aware two-ply search to choose a move.
"""

import copy
import random
import time
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


# ==================== BASIC BOARD UTILITIES ====================

def in_bounds(x: int, y: int, rows: int, cols: int) -> bool:
    return 0 <= x < cols and 0 <= y < rows


def score_cols_for(cols: int) -> List[int]:
    w = 4
    start = max(0, (cols - w) // 2)
    return list(range(start, start + w))


def top_score_row() -> int:
    return 2


def bottom_score_row(rows: int) -> int:
    return rows - 3


def get_opponent(player: str) -> str:
    return "square" if player == "circle" else "circle"


def is_opponent_score_cell(
    x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]
) -> bool:
    if player == "circle":
        return y == bottom_score_row(rows) and x in score_cols
    return y == top_score_row() and x in score_cols


def is_own_score_cell(
    x: int, y: int, player: str, rows: int, cols: int, score_cols: List[int]
) -> bool:
    return is_opponent_score_cell(
        x, y, get_opponent(player), rows, cols, score_cols
    )


# ==================== AUTHORITATIVE MOVE GENERATION ====================
# This mirrors the rules implemented by gameEngine.py.  Keeping the logic
# here avoids importing gameEngine during its own import of student_agent.py.


def river_flow_destinations(
    board,
    rx: int,
    ry: int,
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
    river_push: bool = False,
) -> List[Tuple[int, int]]:
    destinations = []
    visited = set()
    queue = [(rx, ry)]

    while queue:
        x, y = queue.pop(0)
        if (x, y) in visited or not in_bounds(x, y, rows, cols):
            continue
        visited.add((x, y))

        cell = board[y][x]
        if river_push and x == rx and y == ry:
            cell = board[sy][sx]

        if cell is None:
            if not is_opponent_score_cell(x, y, player, rows, cols, score_cols):
                destinations.append((x, y))
            continue

        if getattr(cell, "side", "stone") != "river":
            continue

        if cell.orientation == "horizontal":
            dirs = [(1, 0), (-1, 0)]
        else:
            dirs = [(0, 1), (0, -1)]

        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            while in_bounds(nx, ny, rows, cols):
                if is_opponent_score_cell(nx, ny, player, rows, cols, score_cols):
                    break

                next_cell = board[ny][nx]
                if next_cell is None:
                    destinations.append((nx, ny))
                    nx += dx
                    ny += dy
                    continue

                # When calculating a river push, the source position is
                # treated as the moving piece and is therefore traversable.
                if nx == sx and ny == sy:
                    nx += dx
                    ny += dy
                    continue

                if getattr(next_cell, "side", "stone") == "river":
                    queue.append((nx, ny))
                    break
                break

    result = []
    seen = set()
    for d in destinations:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def compute_valid_targets(
    board,
    sx: int,
    sy: int,
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Dict[str, Any]:
    if not in_bounds(sx, sy, rows, cols):
        return {"moves": set(), "pushes": []}

    piece = board[sy][sx]
    if piece is None or piece.owner != player:
        return {"moves": set(), "pushes": []}

    moves = set()
    pushes = []
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for dx, dy in directions:
        tx, ty = sx + dx, sy + dy
        if not in_bounds(tx, ty, rows, cols):
            continue
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            continue

        target = board[ty][tx]

        if target is None:
            moves.add((tx, ty))

        elif getattr(target, "side", "stone") == "river":
            flow = river_flow_destinations(
                board, tx, ty, sx, sy, player, rows, cols, score_cols
            )
            moves.update(flow)

        else:
            # Stone can push a stone one square.
            if getattr(piece, "side", "stone") == "stone":
                px, py = tx + dx, ty + dy
                if (
                    in_bounds(px, py, rows, cols)
                    and board[py][px] is None
                    and not is_opponent_score_cell(
                        px, py, player, rows, cols, score_cols
                    )
                ):
                    pushes.append(((tx, ty), (px, py)))

            # River can push a stone along the pushed piece's flow.  A river
            # cannot push another river.
            else:
                pushed_player = target.owner
                flow = river_flow_destinations(
                    board,
                    tx,
                    ty,
                    sx,
                    sy,
                    pushed_player,
                    rows,
                    cols,
                    score_cols,
                    river_push=True,
                )
                for dest in flow:
                    if not is_opponent_score_cell(
                        dest[0], dest[1], player, rows, cols, score_cols
                    ):
                        pushes.append(((tx, ty), dest))

    return {"moves": moves, "pushes": pushes}


def generate_all_moves(
    board: List[List[Any]],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> List[Dict[str, Any]]:
    moves = []

    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if piece is None or piece.owner != player:
                continue

            info = compute_valid_targets(board, x, y, player, rows, cols, score_cols)

            for tx, ty in sorted(info["moves"]):
                moves.append(
                    {"action": "move", "from": [x, y], "to": [tx, ty]}
                )

            for (tx, ty), (px, py) in info["pushes"]:
                moves.append(
                    {
                        "action": "push",
                        "from": [x, y],
                        "to": [tx, ty],
                        "pushed_to": [px, py],
                    }
                )

            if piece.side == "stone":
                # A flip is legal only when its resulting river flow is safe.
                for orientation in ("horizontal", "vertical"):
                    temp = copy.deepcopy(board)
                    temp[y][x].side = "river"
                    temp[y][x].orientation = orientation
                    flow = river_flow_destinations(
                        temp, x, y, x, y, player, rows, cols, score_cols
                    )
                    if not any(
                        is_opponent_score_cell(
                            fx, fy, player, rows, cols, score_cols
                        )
                        for fx, fy in flow
                    ):
                        moves.append(
                            {
                                "action": "flip",
                                "from": [x, y],
                                "orientation": orientation,
                            }
                        )
            else:
                moves.append({"action": "flip", "from": [x, y]})

                # Rotation is legal only when the new river orientation is safe.
                new_orientation = (
                    "vertical"
                    if piece.orientation == "horizontal"
                    else "horizontal"
                )
                temp = copy.deepcopy(board)
                temp[y][x].orientation = new_orientation
                flow = river_flow_destinations(
                    temp, x, y, x, y, player, rows, cols, score_cols
                )
                if not any(
                    is_opponent_score_cell(
                        fx, fy, player, rows, cols, score_cols
                    )
                    for fx, fy in flow
                ):
                    moves.append({"action": "rotate", "from": [x, y]})

    return moves


# ==================== MOVE APPLICATION FOR SEARCH ====================


def apply_move(
    board: List[List[Any]],
    move: Dict[str, Any],
    player: str,
    rows: int,
    cols: int,
    score_cols: List[int],
) -> Tuple[bool, Any]:
    """Apply a move to a board copy using the engine's move semantics."""
    action = move.get("action")
    fr = move.get("from")

    if not fr:
        return False, "missing from"

    fx, fy = int(fr[0]), int(fr[1])
    if not in_bounds(fx, fy, rows, cols):
        return False, "out of bounds"

    piece = board[fy][fx]
    if piece is None or piece.owner != player:
        return False, "invalid piece"

    if action == "move":
        to = move.get("to")
        if not to:
            return False, "missing to"
        tx, ty = int(to[0]), int(to[1])
        if not in_bounds(tx, ty, rows, cols):
            return False, "out of bounds"
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            return False, "opponent score"

        if board[ty][tx] is None:
            board[ty][tx] = piece
            board[fy][fx] = None
            return True, board

        pushed = move.get("pushed_to")
        if not pushed:
            return False, "occupied destination"
        px, py = int(pushed[0]), int(pushed[1])
        dx, dy = tx - fx, ty - fy
        if (px, py) != (tx + dx, ty + dy):
            return False, "bad push geometry"
        if not in_bounds(px, py, rows, cols):
            return False, "out of bounds"
        if is_opponent_score_cell(px, py, player, rows, cols, score_cols):
            return False, "opponent score"
        if board[py][px] is not None:
            return False, "push destination occupied"

        board[py][px] = board[ty][tx]
        board[ty][tx] = piece
        board[fy][fx] = None
        return True, board

    if action == "push":
        to = move.get("to")
        pushed = move.get("pushed_to")
        if not to or not pushed:
            return False, "bad push"
        tx, ty = int(to[0]), int(to[1])
        px, py = int(pushed[0]), int(pushed[1])
        if not (
            in_bounds(tx, ty, rows, cols)
            and in_bounds(px, py, rows, cols)
        ):
            return False, "out of bounds"
        if board[ty][tx] is None or board[py][px] is not None:
            return False, "invalid push cells"
        if is_opponent_score_cell(tx, ty, player, rows, cols, score_cols):
            return False, "opponent score"

        pushed_piece = board[ty][tx]
        if is_opponent_score_cell(
            px, py, pushed_piece.owner, rows, cols, score_cols
        ):
            return False, "pushed piece enters opponent score"

        info = compute_valid_targets(
            board, fx, fy, player, rows, cols, score_cols
        )
        if ((tx, ty), (px, py)) not in info["pushes"]:
            return False, "invalid push pair"

        board[py][px] = pushed_piece
        board[ty][tx] = piece
        board[fy][fx] = None

        # The engine flips the pushing river to a stone after a push.
        if piece.side == "river":
            board[ty][tx].side = "stone"
            board[ty][tx].orientation = None
        return True, board

    if action == "flip":
        if piece.side == "stone":
            orientation = move.get("orientation")
            if orientation not in ("horizontal", "vertical"):
                return False, "missing orientation"
            piece.side = "river"
            piece.orientation = orientation
            flow = river_flow_destinations(
                board, fx, fy, fx, fy, player, rows, cols, score_cols
            )
            piece.side = "stone"
            piece.orientation = None
            if any(
                is_opponent_score_cell(x, y, player, rows, cols, score_cols)
                for x, y in flow
            ):
                return False, "unsafe flip"
            piece.side = "river"
            piece.orientation = orientation
            return True, board

        piece.side = "stone"
        piece.orientation = None
        return True, board

    if action == "rotate":
        if piece.side != "river":
            return False, "not a river"
        old = piece.orientation
        piece.orientation = (
            "vertical" if old == "horizontal" else "horizontal"
        )
        flow = river_flow_destinations(
            board, fx, fy, fx, fy, player, rows, cols, score_cols
        )
        if any(
            is_opponent_score_cell(x, y, player, rows, cols, score_cols)
            for x, y in flow
        ):
            piece.orientation = old
            return False, "unsafe rotation"
        return True, board

    return False, "unknown action"


# ==================== POSITIONAL EVALUATION ====================


def count_scoring_stones(
    board, player: str, rows: int, cols: int, score_cols: List[int]
) -> int:
    row = top_score_row() if player == "circle" else bottom_score_row(rows)
    return sum(
        1
        for x in score_cols
        if in_bounds(x, row, rows, cols)
        and board[row][x] is not None
        and board[row][x].owner == player
        and board[row][x].side == "stone"
    )


def reachable_scoring_threats(
    board, player: str, rows: int, cols: int, score_cols: List[int]
) -> int:
    """Number of stones that can enter the owner's score area in one move."""
    total = 0
    for y in range(rows):
        for x in range(cols):
            piece = board[y][x]
            if not piece or piece.owner != player:
                continue
            if piece.side == "river":
                if is_own_score_cell(x, y, player, rows, cols, score_cols):
                    total += 1
                continue
            if is_own_score_cell(x, y, player, rows, cols, score_cols):
                continue

            info = compute_valid_targets(
                board, x, y, player, rows, cols, score_cols
            )
            if any(
                is_own_score_cell(tx, ty, player, rows, cols, score_cols)
                for tx, ty in info["moves"]
            ):
                total += 1
                continue
            if any(
                is_own_score_cell(px, py, player, rows, cols, score_cols)
                for _, (px, py) in info["pushes"]
            ):
                total += 1
    return total


def distance_to_opponent_score(
    x: int, y: int, player: str, rows: int, score_cols: List[int]
) -> int:
    # The winning/scoring area is the area occupied by the opponent at the
    # beginning: top for Circle and bottom for Square.
    target_row = top_score_row() if player == "circle" else bottom_score_row(rows)
    row_distance = abs(target_row - y)
    if x in score_cols:
        col_distance = 0
    else:
        col_distance = min(abs(x - c) for c in score_cols)
    return row_distance + col_distance


def evaluate_position(
    board, player: str, rows: int, cols: int, score_cols: List[int]
) -> float:
    """Fast positional evaluation from the current player's perspective."""
    opponent = get_opponent(player)
    value = 0.0

    my_scoring = count_scoring_stones(board, player, rows, cols, score_cols)
    opp_scoring = count_scoring_stones(
        board, opponent, rows, cols, score_cols
    )

    # Four stones wins.  These weights dominate all secondary features.
    value += 2200.0 * my_scoring
    value -= 2350.0 * opp_scoring

    max_distance = rows + cols
    my_stones = 0
    opp_stones = 0

    for y in range(rows):
        for x in range(cols):
            p = board[y][x]
            if not p:
                continue

            if p.owner == player and p.side == "stone":
                my_stones += 1
                d = distance_to_opponent_score(
                    x, y, player, rows, score_cols
                )
                value += 65.0 * (max_distance - d) / max_distance

                if is_own_score_cell(
                    x, y, player, rows, cols, score_cols
                ):
                    value += 900.0

            elif p.owner == opponent and p.side == "stone":
                opp_stones += 1
                d = distance_to_opponent_score(
                    x, y, opponent, rows, score_cols
                )
                value -= 65.0 * (max_distance - d) / max_distance

                if is_own_score_cell(
                    x, y, opponent, rows, cols, score_cols
                ):
                    value -= 950.0

            elif p.owner == player and p.side == "river":
                # Small mobility bonus for our rivers.  Count open neighbours
                # only; full river-flow analysis is reserved for move search.
                open_neighbours = 0
                if p.orientation == "horizontal":
                    dirs = ((1, 0), (-1, 0))
                else:
                    dirs = ((0, 1), (0, -1))
                for dx, dy in dirs:
                    nx, ny = x + dx, y + dy
                    if in_bounds(nx, ny, rows, cols) and board[ny][nx] is None:
                        open_neighbours += 1
                value += 1.5 * open_neighbours

            elif p.owner == opponent and p.side == "river":
                value -= 0.7

    # Preserve material symmetry while making actual score progress primary.
    value += 0.25 * (my_stones - opp_stones)
    return value


# ==================== STUDENT AGENT ====================


class BaseAgent(ABC):
    def __init__(self, player: str):
        self.player = player
        self.opponent = get_opponent(player)

    @abstractmethod
    def choose(
        self,
        board: List[List[Any]],
        rows: int,
        cols: int,
        score_cols: List[int],
        current_player_time: float,
        opponent_time: float,
    ) -> Optional[Dict[str, Any]]:
        pass


class StudentAgent(BaseAgent):
    """A time-aware tactical + shallow-search player."""

    def __init__(self, player: str):
        super().__init__(player)
        self.random = random.Random(1009 if player == "circle" else 2027)

    def generate_all_moves(self, board, rows, cols, score_cols):
        return generate_all_moves(
            board, self.player, rows, cols, score_cols
        )

    def simulate_move(self, board, move, rows, cols, score_cols):
        b = copy.deepcopy(board)
        ok, result = apply_move(
            b, move, self.player, rows, cols, score_cols
        )
        return ok, result

    def _is_winning_board(self, board, rows, cols, score_cols, player):
        return count_scoring_stones(
            board, player, rows, cols, score_cols
        ) >= 4

    def _move_key(self, move):
        return (
            move.get("action", ""),
            tuple(move.get("from", [])),
            tuple(move.get("to", [])),
            tuple(move.get("pushed_to", [])),
            move.get("orientation", ""),
        )

    def _quick_move_score(self, board, move, rows, cols, score_cols):
        """Cheap ordering heuristic used before the deeper search."""
        ok, new_board = self.simulate_move(
            board, move, rows, cols, score_cols
        )
        if not ok:
            return -10**9

        before = evaluate_position(
            board, self.player, rows, cols, score_cols
        )
        after = evaluate_position(
            new_board, self.player, rows, cols, score_cols
        )
        score = after - before

        action = move.get("action")
        if action == "push":
            score += 8.0
        elif action == "flip":
            score += 1.0
        elif action == "rotate":
            score += 1.5
        return score

    def _ordered_moves(
        self, board, moves, rows, cols, score_cols, limit=None
    ):
        scored = []
        for move in moves:
            s = self._quick_move_score(
                board, move, rows, cols, score_cols
            )
            scored.append((s, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        if limit is not None:
            scored = scored[:limit]
        return [move for _, move in scored]

    def _opponent_reply_value(
        self, board, rows, cols, score_cols, deadline
    ):
        """Best reply value for the opponent, with a small move beam."""
        if time.perf_counter() >= deadline:
            return evaluate_position(
                board, self.player, rows, cols, score_cols
            )

        replies = generate_all_moves(
            board, self.opponent, rows, cols, score_cols
        )
        if not replies:
            return evaluate_position(
                board, self.player, rows, cols, score_cols
            )

        # The opponent only needs to consider their most dangerous replies.
        scored = []
        for reply in replies:
            if time.perf_counter() >= deadline:
                break
            b = copy.deepcopy(board)
            ok, result = apply_move(
                b, reply, self.opponent, rows, cols, score_cols
            )
            if not ok:
                continue
            v = evaluate_position(
                b, self.player, rows, cols, score_cols
            )
            scored.append((v, reply, b))

        if not scored:
            return evaluate_position(
                board, self.player, rows, cols, score_cols
            )

        # Opponent minimizes our score.
        scored.sort(key=lambda item: item[0])
        return scored[0][0]

    def choose(
        self,
        board: List[List[Any]],
        rows: int,
        cols: int,
        score_cols: List[int],
        current_player_time: float,
        opponent_time: float,
    ) -> Optional[Dict[str, Any]]:
        start = time.perf_counter()

        moves = self.generate_all_moves(board, rows, cols, score_cols)
        if not moves:
            return None

        # Never spend meaningful search time when the clock is almost empty.
        if current_player_time <= 0.75:
            return self._ordered_moves(
                board, moves, rows, cols, score_cols, limit=1
            )[0]

        # Tactical priority 1: immediate winning move.
        for move in moves:
            ok, new_board = self.simulate_move(
                board, move, rows, cols, score_cols
            )
            if ok and self._is_winning_board(
                new_board, rows, cols, score_cols, self.player
            ):
                return move
            if time.perf_counter() - start > min(0.25, current_player_time * 0.05):
                break

        # Tactical priority 2: if the opponent has a one-move win, prefer a
        # move that removes all immediate opponent winning moves.
        opponent_immediate = []
        opp_moves_now = generate_all_moves(
            board, self.opponent, rows, cols, score_cols
        )
        for om in opp_moves_now:
            b = copy.deepcopy(board)
            ok, result = apply_move(
                b, om, self.opponent, rows, cols, score_cols
            )
            if ok and self._is_winning_board(
                b, rows, cols, score_cols, self.opponent
            ):
                opponent_immediate.append(om)

        if opponent_immediate:
            safe_blocks = []
            for move in moves:
                if time.perf_counter() >= start + min(0.45, current_player_time * 0.08):
                    break
                ok, b = self.simulate_move(
                    board, move, rows, cols, score_cols
                )
                if not ok:
                    continue
                replies = generate_all_moves(
                    b, self.opponent, rows, cols, score_cols
                )
                can_win = False
                for reply in replies:
                    rb = copy.deepcopy(b)
                    rok, rr = apply_move(
                        rb, reply, self.opponent, rows, cols, score_cols
                    )
                    if rok and self._is_winning_board(
                        rb, rows, cols, score_cols, self.opponent
                    ):
                        can_win = True
                        break
                if not can_win:
                    safe_blocks.append(move)

            if safe_blocks:
                moves = safe_blocks

        # Main search.  Keep a small beam to stay comfortably inside the
        # one-minute total game clock while still looking beyond one move.
        time_left = max(0.0, current_player_time - (time.perf_counter() - start))
        if time_left < 1.0:
            return self._ordered_moves(
                board, moves, rows, cols, score_cols, limit=1
            )[0]

        beam = 10 if time_left > 5.0 else 7
        opponent_beam = 7 if time_left > 5.0 else 5
        candidates = self._ordered_moves(
            board, moves, rows, cols, score_cols, limit=beam
        )

        deadline = time.perf_counter() + min(
            0.35, max(0.08, time_left * 0.035)
        )

        best_move = candidates[0]
        best_value = -float("inf")

        for move in candidates:
            if time.perf_counter() >= deadline:
                break

            ok, new_board = self.simulate_move(
                board, move, rows, cols, score_cols
            )
            if not ok:
                continue

            # Winning moves are already checked, but keep this guard here.
            if self._is_winning_board(
                new_board, rows, cols, score_cols, self.player
            ):
                return move

            replies = generate_all_moves(
                new_board, self.opponent, rows, cols, score_cols
            )
            if not replies:
                value = evaluate_position(
                    new_board, self.player, rows, cols, score_cols
                ) + 100.0
            else:
                ordered_replies = self._ordered_opponent_replies(
                    new_board,
                    replies,
                    rows,
                    cols,
                    score_cols,
                    opponent_beam,
                )
                value = float("inf")
                for reply in ordered_replies:
                    if time.perf_counter() >= deadline:
                        break
                    rb = copy.deepcopy(new_board)
                    rok, result = apply_move(
                        rb, reply, self.opponent, rows, cols, score_cols
                    )
                    if not rok:
                        continue
                    reply_value = evaluate_position(
                        rb, self.player, rows, cols, score_cols
                    )
                    if self._is_winning_board(
                        rb, rows, cols, score_cols, self.opponent
                    ):
                        reply_value -= 10000.0
                    value = min(value, reply_value)

                if value == float("inf"):
                    value = evaluate_position(
                        new_board, self.player, rows, cols, score_cols
                    )

            # Tiny deterministic tie-breaker; it avoids repetitive arbitrary
            # choices without changing the strategic evaluation.
            value += self.random.random() * 1e-5
            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def _ordered_opponent_replies(
        self,
        board,
        replies,
        rows,
        cols,
        score_cols,
        limit,
    ):
        scored = []
        for reply in replies:
            b = copy.deepcopy(board)
            ok, result = apply_move(
                b, reply, self.opponent, rows, cols, score_cols
            )
            if not ok:
                continue
            # From our perspective, a dangerous opponent reply has a low
            # resulting evaluation, so sort ascending.
            score = evaluate_position(
                b, self.player, rows, cols, score_cols
            )
            scored.append((score, reply))
        scored.sort(key=lambda item: item[0])
        return [reply for _, reply in scored[:limit]]

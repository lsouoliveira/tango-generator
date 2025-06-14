from enum import Enum
from typing import List, Optional
import random
from PIL import Image, ImageDraw

GRID_BORDER_COLOR = "#e6e6e6"
GRID_TILE_SIZE = 128
OUTPUT_IMAGE_SIZE = (GRID_TILE_SIZE * 6, GRID_TILE_SIZE * 6)
SUN_IMAGE_HEIGHT = 128
MOON_IMAGE_HEIGHT = SUN_IMAGE_HEIGHT // 2
X_IMAGE_HEIGHT = SUN_IMAGE_HEIGHT // 4
EQUAL_SIGN_IMAGE_HEIGHT = SUN_IMAGE_HEIGHT // 4


def resize_to_fit(image, height):
    new_width = int(height * image.width / image.height)

    return image.resize((new_width, height), Image.LANCZOS)


sun_image = resize_to_fit(Image.open("assets/sun.png"), SUN_IMAGE_HEIGHT)
moon_image = resize_to_fit(Image.open("assets/moon.png"), MOON_IMAGE_HEIGHT)
x_image = resize_to_fit(Image.open("assets/x.png"), X_IMAGE_HEIGHT)
equal_sign_image = resize_to_fit(
    Image.open("assets/equal_sign.png"), EQUAL_SIGN_IMAGE_HEIGHT
)


class Piece(Enum):
    SUN = 0
    MOON = 1


PIECE_TO_SYMBOL = {Piece.SUN: "SUN", Piece.MOON: "MOON"}

PIECE_TO_IMAGE = {Piece.SUN: sun_image, Piece.MOON: moon_image}


def pretty_format_puzzle(puzzle: "Puzzle") -> str:
    output = ""

    for i in range(puzzle.rows_count()):
        for j in range(puzzle.cols_count()):
            symbol = puzzle.peek(i, j)

            if not symbol:
                output += "_"
            else:
                output += PIECE_TO_SYMBOL[symbol]

            if j < 5:
                output += ", "

        output += "\n"

    return output


def opposite_piece(piece: Piece) -> Piece:
    if piece == Piece.SUN:
        return Piece.MOON
    elif piece == Piece.MOON:
        return Piece.SUN
    else:
        raise ValueError("Unknown piece type")


INVALID_PUZZLE = [
    [Piece.SUN, Piece.SUN, None, None, Piece.SUN, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [Piece.SUN, None, None, None, None, None],
    [None, None, None, None, None, None],
    [Piece.SUN, None, None, None, None, None],
]


def empty_puzzle():
    return [
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
        [None, None, None, None, None, None],
    ]


def group_sequence(sequence: List[Optional[Piece]]) -> dict:
    d = {}

    for symbol in sequence:
        if not symbol:
            continue

        d[symbol] = d.get(symbol, 0) + 1

    return d


class ConnectionType(Enum):
    EQUAL = 0
    DIFFERENT = 1


class Direction(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class Connection:
    from_: tuple[int, int]
    to: tuple[int, int]
    connection_type: ConnectionType
    puzzle: "Puzzle"

    def __init__(
        self,
        from_: tuple[int, int],
        to: tuple[int, int],
        puzzle: "Puzzle",
        connection_type: ConnectionType,
    ):
        self.from_ = from_
        self.to = to
        self.puzzle = puzzle
        self.connection_type = connection_type

    def direction(self) -> Direction:
        if self.from_[0] < self.to[0]:
            return Direction.DOWN
        elif self.from_[0] > self.to[0]:
            return Direction.UP
        elif self.from_[1] < self.to[1]:
            return Direction.RIGHT
        elif self.from_[1] > self.to[1]:
            return Direction.LEFT
        else:
            raise ValueError("Invalid connection direction")

    def is_valid(self) -> bool:
        raise NotImplementedError


class EqualConnection(Connection):
    def __init__(self, from_: tuple[int, int], to: tuple[int, int], puzzle: "Puzzle"):
        super().__init__(from_, to, puzzle, ConnectionType.EQUAL)

    def is_valid(self) -> bool:
        piece_from = self.puzzle.peek(self.from_[0], self.from_[1])
        piece_to = self.puzzle.peek(self.to[0], self.to[1])

        if not piece_from or not piece_to:
            return True

        return piece_from == piece_to


class DifferentConnection(Connection):
    def __init__(self, from_: tuple[int, int], to: tuple[int, int], puzzle: "Puzzle"):
        super().__init__(from_, to, puzzle, ConnectionType.DIFFERENT)

    def is_valid(self) -> bool:
        piece_from = self.puzzle.peek(self.from_[0], self.from_[1])
        piece_to = self.puzzle.peek(self.to[0], self.to[1])

        if not piece_from or not piece_to:
            return True

        return piece_from != piece_to


class Puzzle:
    grid: List[List[Optional[Piece]]]

    connections: dict[tuple[int, int], List[Connection]]

    def __init__(self, grid):
        self.grid = grid
        self.connections = {}

    def peek(self, row, col) -> Optional[Piece]:
        return self.grid[row][col]

    def place_piece(self, row: int, col: int, piece_type: Optional[Piece]):
        self.grid[row][col] = piece_type

    def completed(self) -> bool:
        pass

    def is_valid(self) -> bool:
        rows = [self.grid[i] for i in range(self.rows_count())]
        columns = [
            [self.grid[i][j] for i in range(self.rows_count())]
            for j in range(self.cols_count())
        ]
        sequences = rows + columns
        are_connections_valid = self._are_connections_valid()

        return (
            all(map(lambda x: self._are_values_valid(x), sequences))
            and are_connections_valid
        )

    def add_connection(
        self,
        from_: tuple[int, int],
        to: tuple[int, int],
        connection_type: ConnectionType,
    ):
        if from_ not in self.connections:
            self.connections[from_] = []

        if to not in self.connections:
            self.connections[to] = []

        from_connection = self.build_connection(from_, to, connection_type)
        to_connection = self.build_connection(to, from_, connection_type)

        self.connections[from_].append(from_connection)
        self.connections[to].append(to_connection)

    def build_connection(
        self,
        from_: tuple[int, int],
        to: tuple[int, int],
        connection_type: ConnectionType,
    ) -> Connection:
        if connection_type == ConnectionType.EQUAL:
            return EqualConnection(from_, to, self)
        elif connection_type == ConnectionType.DIFFERENT:
            return DifferentConnection(from_, to, self)
        else:
            raise ValueError("Unknown connection type")

    def get_connection(
        self, from_: tuple[int, int], to: tuple[int, int]
    ) -> Optional[Connection]:
        if from_ in self.connections:
            for connection in self.connections[from_]:
                if connection.to == to:
                    return connection
        return None

    def get_neighbours(self, row: int, col: int) -> List[tuple[int, int]]:
        neighbours = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]

        return list(
            filter(lambda x: not self._is_out_of_bounds(x[0], x[1]), neighbours)
        )

    def rows_count(self) -> int:
        return len(self.grid)

    def cols_count(self) -> int:
        return len(self.grid[0])

    def get_column(self, index: int) -> List[Optional[Piece]]:
        return [self.grid[i][index] for i in range(self.rows_count())]

    def get_row(self, index: int) -> List[Optional[Piece]]:
        return self.grid[index]

    def filled_cells_count(self):
        filled_cells = 0

        for i in range(self.rows_count()):
            for j in range(self.cols_count()):
                if self.grid[i][j]:
                    filled_cells += 1

        return filled_cells

    def _are_values_valid(self, values: List[Optional[Piece]]):
        seen = {}
        sequence = 0
        last_seen = None

        for symbol in values:
            if not symbol:
                last_seen = None
                continue

            if symbol == last_seen:
                sequence += 1
            else:
                sequence = 1

            if sequence >= 3:
                return False

            seen[symbol] = seen.get(symbol, 0) + 1
            last_seen = symbol

        for value in seen.values():
            if value > 3:
                return False

        return True

    def _are_connections_valid(self) -> bool:
        for connections in self.connections.values():
            for connection in connections:
                if not connection.is_valid():
                    return False

        return True

    def _is_out_of_bounds(self, row: int, col: int) -> bool:
        return (
            row < 0 or row >= self.rows_count() or col < 0 or col >= self.cols_count()
        )

    def __repr__(self):
        return pretty_format_puzzle(self)


class PuzzleGenerator:
    """
    Generates a random puzzle with a given number of connections.
    """

    connections: int

    def __init__(self, connections: int = 3):
        self.connections = connections

    def generate(self) -> Puzzle:
        cell = 0
        stack = []
        puzzle = Puzzle(empty_puzzle())

        self._add_connections(puzzle)

        stack.append(self._symbols())

        iterations = 0

        while len(stack) and cell < 36:
            iterations += 1

            symbols = stack[-1]

            if len(symbols) == 0:
                stack.pop()
                cell -= 1
                continue

            next_piece = symbols[0]

            symbols.pop(0)

            puzzle.place_piece(cell // 6, cell % 6, next_piece)

            if puzzle.is_valid():
                cell += 1
                stack.append(self._symbols())
            else:
                puzzle.place_piece(cell // 6, cell % 6, None)

        return puzzle

    def _symbols(self):
        symbols = [Piece.SUN, Piece.MOON]

        random.shuffle(symbols)

        return symbols

    def _add_connections(self, puzzle: Puzzle):
        i = self.connections

        while i > 0:
            x, y = random.randint(0, 5), random.randint(0, 5)
            neighbours = puzzle.get_neighbours(x, y)

            if len(neighbours) == 0:
                continue

            neighbour = random.choice(neighbours)

            if puzzle.get_connection((x, y), neighbour):
                continue

            connection_type = random.choice(list(ConnectionType))

            puzzle.add_connection((x, y), neighbour, connection_type)

            i -= 1


class SolverStrategy:
    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> Puzzle:
        raise NotImplementedError


class AddComplementStrategy(SolverStrategy):
    """
    Checks if there is already 3 symbols equal in the current column or row.
    If true, fills the cell with the opposite symbol.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        row_groups = group_sequence(puzzle.get_row(row))

        for k, v in row_groups.items():
            if v >= 3:
                opposite_symbol = opposite_piece(k)

                puzzle.place_piece(row, col, opposite_symbol)

                return True

        col_groups = group_sequence(puzzle.get_column(col))

        for k, v in col_groups.items():
            if v >= 3:
                opposite_symbol = opposite_piece(k)

                puzzle.place_piece(row, col, opposite_symbol)

                return True

        return False


class AdjacentStrategy(SolverStrategy):
    """
    Check if a symbol appears twice in a row on the sides.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        if (
            col - 2 >= 0
            and puzzle.peek(row, col - 2) != None
            and puzzle.peek(row, col - 2) == puzzle.peek(row, col - 1)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row, col - 1)))

            return True

        if (
            col + 2 <= 5
            and puzzle.peek(row, col + 2) != None
            and puzzle.peek(row, col + 2) == puzzle.peek(row, col + 1)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row, col + 1)))

            return True

        if (
            row - 2 >= 0
            and puzzle.peek(row - 2, col) != None
            and puzzle.peek(row - 2, col) == puzzle.peek(row - 1, col)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row - 1, col)))

            return True

        if (
            row + 2 <= 5
            and puzzle.peek(row + 2, col) != None
            and puzzle.peek(row + 2, col) == puzzle.peek(row + 1, col)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row + 1, col)))

            return True


class EdgeSequenceStrategy(SolverStrategy):
    """
    Check if a symbol appears twice on the edge.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        if (
            col == 5
            and puzzle.peek(row, 0) != None
            and puzzle.peek(row, 0) == puzzle.peek(row, 1)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row, 1)))

            return True

        if (
            col == 0
            and puzzle.peek(row, 4) != None
            and puzzle.peek(row, 4) == puzzle.peek(row, 5)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(row, 4)))

            return True

        if (
            row == 5
            and puzzle.peek(0, col)
            and puzzle.peek(0, col) == puzzle.peek(1, col)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(1, col)))

            return True

        if (
            row == 0
            and puzzle.peek(4, col)
            and puzzle.peek(4, col) == puzzle.peek(5, col)
        ):
            puzzle.place_piece(row, col, opposite_piece(puzzle.peek(4, col)))

            return True

        return False


class AdjacentToEqualStrategy(SolverStrategy):
    """
    Checks if there is an adjacent cell with an equal connection and with at
    least one piece in it.
    If true, fills the cell with the opposite symbol.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        neighbours = puzzle.get_neighbours(row, col)

        for neighbour in neighbours:
            for connection in puzzle.connections.get((row, col), []):
                if connection.connection_type != ConnectionType.EQUAL:
                    continue

                if connection.to == (row, col):
                    continue

                piece_from = puzzle.peek(row, col)
                piece_to = puzzle.peek(neighbour[0], neighbour[1])

                if not piece_from and not piece_to:
                    continue

                piece = piece_from if piece_from else piece_to

                if (
                    neighbour[0] < row
                    or neighbour[0] > row
                    and connection.direction()
                    in (
                        Direction.LEFT,
                        Direction.RIGHT,
                    )
                ):
                    continue

                if (
                    neighbour[1] < col
                    or neighbour[1] > col
                    and connection.direction()
                    in (
                        Direction.UP,
                        Direction.DOWN,
                    )
                ):
                    continue

                opposite_symbol = opposite_piece(piece)
                puzzle.place_piece(row, col, opposite_symbol)

                return True

        return False


class FillEqualStrategy(SolverStrategy):
    """
    Check if there is an equal connection for the target cell. If the to
    cell is filled, fill the same symbol in the to cell.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        connections = puzzle.connections.get((row, col), [])

        for connection in connections:
            if connection.connection_type != ConnectionType.EQUAL:
                continue

            piece_to = puzzle.peek(connection.to[0], connection.to[1])

            if not piece_to:
                continue

            puzzle.place_piece(row, col, piece_to)

            return True

        return False


class FillDifferentStrategy(SolverStrategy):
    """
    Check if there is a different connection for the target cell. If the to
    cell is filled, fill the opposite symbol in the to cell.
    """

    def apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        connections = puzzle.connections.get((row, col), [])

        for connection in connections:
            if connection.connection_type != ConnectionType.DIFFERENT:
                continue

            piece_to = puzzle.peek(connection.to[0], connection.to[1])

            if not piece_to:
                continue

            opposite_symbol = opposite_piece(piece_to)
            puzzle.place_piece(row, col, opposite_symbol)

            return True

        return False


class PuzzleSolver:
    puzzle: Puzzle
    strategies: List[SolverStrategy]

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

        self.strategies = [
            AddComplementStrategy(),
            AdjacentStrategy(),
            EdgeSequenceStrategy(),
            AdjacentToEqualStrategy(),
            FillEqualStrategy(),
            FillDifferentStrategy(),
        ]

        random.shuffle(self.strategies)

    def can_solve_cell(self, row: int, col: int) -> bool:
        for strategy in self.strategies:
            if strategy.apply_at(self.puzzle, row, col) and self.puzzle.is_valid():
                return True

        return False


class ProblemBuilder:
    min_pieces: int
    connections: int

    def __init__(self, min_pieces=4, connections=3):
        self.min_pieces = min_pieces
        self.connections = connections

    def build(self) -> Puzzle:
        puzzle = PuzzleGenerator(connections=self.connections).generate()

        while puzzle.filled_cells_count() < 36:
            puzzle = PuzzleGenerator(connections=self.connections).generate()

        solver = PuzzleSolver(puzzle)

        cells = [
            (i, j)
            for i in range(puzzle.rows_count())
            for j in range(puzzle.cols_count())
        ]
        remaining_pieces = 36

        random.shuffle(cells)

        for i, j in cells:
            symbol = puzzle.peek(i, j)

            puzzle.place_piece(i, j, None)

            if not solver.can_solve_cell(i, j):
                puzzle.place_piece(i, j, symbol)
            else:
                puzzle.place_piece(i, j, None)
                remaining_pieces -= 1

            if remaining_pieces <= self.min_pieces:
                break

        return puzzle


class ProblemBuilderByAlternatingPieces:
    min_pieces: int
    connections: int

    def __init__(self, min_pieces=4, connections=3):
        self.min_pieces = min_pieces
        self.connections = connections

    def build(self) -> Puzzle:
        puzzle = PuzzleGenerator(connections=self.connections).generate()

        while puzzle.filled_cells_count() < 36:
            puzzle = PuzzleGenerator(connections=self.connections).generate()

        solver = PuzzleSolver(puzzle)

        moon_cells = [
            (i, j)
            for i in range(puzzle.rows_count())
            for j in range(puzzle.cols_count())
            if puzzle.peek(i, j) == Piece.MOON
        ]

        random.shuffle(moon_cells)

        sun_cells = [
            (i, j)
            for i in range(puzzle.rows_count())
            for j in range(puzzle.cols_count())
            if puzzle.peek(i, j) == Piece.SUN
        ]

        random.shuffle(sun_cells)

        cells = []

        for i in range(18):
            if random.randint(0, 1) == 0:
                cells.append(moon_cells[i])
                cells.append(sun_cells[i])
            else:
                cells.append(sun_cells[i])
                cells.append(moon_cells[i])

        remaining_pieces = 36

        for i, j in cells:
            symbol = puzzle.peek(i, j)

            puzzle.place_piece(i, j, None)

            if not solver.can_solve_cell(i, j):
                puzzle.place_piece(i, j, symbol)
            else:
                puzzle.place_piece(i, j, None)
                remaining_pieces -= 1

            if remaining_pieces <= self.min_pieces:
                break

        return puzzle


def generate_puzzle_image(puzzle: Puzzle) -> Image:
    image = Image.new("RGB", OUTPUT_IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)

    for i in range(7):
        if i < 6:
            draw.line(
                ((0, i * GRID_TILE_SIZE), (image.width, i * GRID_TILE_SIZE)),
                fill=GRID_BORDER_COLOR,
            )
        else:
            draw.line(
                [(0, i * GRID_TILE_SIZE - 1), (image.width, i * GRID_TILE_SIZE - 1)],
                fill=GRID_BORDER_COLOR,
            )

    for i in range(7):
        if i < 6:
            draw.line(
                [(i * GRID_TILE_SIZE, 0), (i * GRID_TILE_SIZE, image.height)],
                fill=GRID_BORDER_COLOR,
            )
        else:
            draw.line(
                [(i * GRID_TILE_SIZE - 1, 0), (i * GRID_TILE_SIZE - 1, image.height)],
                fill=GRID_BORDER_COLOR,
            )

    for i in range(puzzle.rows_count()):
        for j in range(puzzle.cols_count()):
            symbol = puzzle.peek(i, j)

            if not symbol:
                continue

            symbol_image = PIECE_TO_IMAGE[symbol]

            image.paste(
                symbol_image,
                (
                    j * GRID_TILE_SIZE + GRID_TILE_SIZE // 2 - symbol_image.width // 2,
                    i * GRID_TILE_SIZE + GRID_TILE_SIZE // 2 - symbol_image.height // 2,
                ),
                symbol_image,
            )

    drawn_connections = set()

    dir = {
        Direction.UP: (0.5, 0),
        Direction.DOWN: (0.5, 1),
        Direction.LEFT: (0, 0.5),
        Direction.RIGHT: (1, 0.5),
    }

    for from_, connections in puzzle.connections.items():
        for connection in connections:
            if (from_, connection.to) in drawn_connections or (
                connection.to,
                from_,
            ) in drawn_connections:
                continue

            drawn_connections.add((from_, connection.to))

            anchor_x, anchor_y = dir[connection.direction()]
            row, col = from_[0], from_[1]

            tile_position = (
                col * GRID_TILE_SIZE + GRID_TILE_SIZE * anchor_x,
                row * GRID_TILE_SIZE + GRID_TILE_SIZE * anchor_y,
            )

            if connection.connection_type == ConnectionType.EQUAL:
                image.paste(
                    equal_sign_image,
                    (
                        int(tile_position[0] - equal_sign_image.width // 2),
                        int(tile_position[1] - equal_sign_image.height // 2),
                    ),
                    equal_sign_image,
                )
            elif connection.connection_type == ConnectionType.DIFFERENT:
                image.paste(
                    x_image,
                    (
                        int(tile_position[0] - x_image.width // 2),
                        int(tile_position[1] - x_image.height // 2),
                    ),
                    x_image,
                )

    return image


def symbol_balance(puzzle: Puzzle) -> int:
    d = {Piece.SUN: 0, Piece.MOON: 0}

    for i in range(puzzle.rows_count()):
        for j in range(puzzle.cols_count()):
            symbol = puzzle.peek(i, j)

            if symbol:
                d[symbol] += 1

    return abs(d[Piece.SUN] - d[Piece.MOON])


def generate_random_puzzle(min_pieces=4, max_iterations=0, connections=3) -> Puzzle:
    best_puzzle = ProblemBuilder(min_pieces, connections).build()

    for _ in range(max_iterations):
        puzzle = ProblemBuilder(min_pieces, connections).build()
        balance_ratio = symbol_balance(puzzle) / puzzle.filled_cells_count()

        if (
            puzzle.filled_cells_count() <= best_puzzle.filled_cells_count()
            and balance_ratio <= 0.11
        ):
            best_puzzle = puzzle

    return best_puzzle


def generate_random_puzzle_by_alternating_pieces(
    min_pieces=4, max_iterations=0, connections=3
) -> Puzzle:
    best_puzzle = ProblemBuilderByAlternatingPieces(min_pieces, connections).build()

    for _ in range(max_iterations):
        puzzle = ProblemBuilderByAlternatingPieces(min_pieces, connections).build()

        if puzzle.filled_cells_count() <= best_puzzle.filled_cells_count():
            best_puzzle = puzzle

    return best_puzzle


def main():
    puzzle = generate_random_puzzle_by_alternating_pieces(
        min_pieces=4, max_iterations=200, connections=10
    )

    image = generate_puzzle_image(puzzle)

    image.show()
    image.save("output.png")


if __name__ == "__main__":
    main()

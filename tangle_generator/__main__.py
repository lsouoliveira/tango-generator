from enum import Enum
from typing import List, Optional
import random
from PIL import Image, ImageDraw

GRID_BORDER_COLOR = "#e6e6e6"
GRID_TILE_SIZE = 128
OUTPUT_IMAGE_SIZE = (GRID_TILE_SIZE * 6, GRID_TILE_SIZE * 6)
SUN_IMAGE_HEIGHT = 128
MOON_IMAGE_HEIGHT = SUN_IMAGE_HEIGHT // 2

def resize_to_fit(image, height):
    new_width = int(height * image.width / image.height)

    return image.resize((new_width, height), Image.LANCZOS)

sun_image = resize_to_fit(Image.open("assets/sun.png"), SUN_IMAGE_HEIGHT)
moon_image = resize_to_fit(Image.open("assets/moon.png"), MOON_IMAGE_HEIGHT)

class Piece(Enum):
    SUN = 0
    MOON = 1

PIECE_TO_SYMBOL = {
    Piece.SUN: "SUN",
    Piece.MOON: "MOON"
}

PIECE_TO_IMAGE = {
    Piece.SUN: sun_image,
    Piece.MOON: moon_image
}

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
        [None, None, None, None, None, None]
    ]

def group_sequence(sequence: List[Optional[Piece]]) -> dict:
    d = {}

    for symbol in sequence:
        if not symbol:
            continue

        d[symbol] = d.get(symbol, 0) + 1

    return d


class Puzzle:
    grid: List[List[Optional[Piece]]]

    def __init__(self, grid):
        self.grid = grid

    def peek(self, row, col) -> Optional[Piece]:
        return self.grid[row][col]

    def place_piece(self, row: int, col: int, piece_type: Optional[Piece]) -> Piece:
        self.grid[row][col] = piece_type

    def completed(self) -> bool:
        pass

    def is_valid(self) -> bool:
        rows = [self.grid[i] for i in range(self.rows_count())]
        columns = [[self.grid[i][j] for i in range(self.rows_count())] for j in range(self.cols_count())]
        sequences = rows + columns

        return all(map(lambda x: self._are_values_valid(x), sequences))

    def rows_count(self) -> int:
        return len(self.grid)

    def cols_count(self) -> int:
        return len(self.grid[0])

    def get_column(self, index: int) -> List[Optional[Piece]]:
        return [self.grid[i][index] for i in range(self.rows_count())]

    def get_row(self, index: int) -> List[Optional[Piece]]:
        return self.grid[index]

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

    def __repr__(self):
        return pretty_format_puzzle(self)

class PuzzleGenerator:
    rows: int
    cols: int

    def __init__(self, rows: int = 3, cols: int = 3):
        self.rows = rows
        self.cols = cols

    def generate(self) -> Puzzle:
        cell = 0
        stack = []
        puzzle = Puzzle(empty_puzzle())

        stack.append(self._symbols())

        while len(stack) and cell < 36:
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

class SolverStrategy:
    def can_apply_at(self, puzzle: Puzzle, row: int, col: int) -> Puzzle:
        raise NotImplementedError

class AddComplementStrategy(SolverStrategy):
    """
    Checks if there is already 3 symbols equal in the current column or row.
    If true, fills the cell with the opposite symbol.
    """

    def can_apply_at(self, puzzle: Puzzle, row: int, col: int) -> bool:
        row_groups = group_sequence(puzzle.get_row(row))

        for v in row_groups.values():
            if v >= 3:
                return True

        col_groups = group_sequence(puzzle.get_column(col))

        for v in col_groups.values():
            if v >= 3:
                return True

        return False


class PuzzleSolver:
    puzzle: Puzzle
    strategies: List[SolverStrategy]

    def __init__(self, puzzle: Puzzle):
        self.puzzle = puzzle

        self.strategies = [
            AddComplementStrategy()
        ]

    def can_solve_cell(self, row: int, col: int) -> bool:
        for strategy in self.strategies:
            if strategy.can_apply_at(self.puzzle, row, col):
                return True

        return False

class ProblemBuilder:
    def build(self) -> Puzzle:
        puzzle = PuzzleGenerator().generate()
        solver = PuzzleSolver(puzzle)

        cells = [(i, j) for i in range(puzzle.rows_count()) for j in range(puzzle.cols_count())]

        random.shuffle(cells)

        for (i, j) in cells:
            symbol = puzzle.peek(i, j)

            puzzle.place_piece(i, j, None)

            if not solver.can_solve_cell(i, j):
                puzzle.place_piece(i, j, symbol)

        return puzzle

def generate_puzzle_image(puzzle: Puzzle) -> Image:
    image = Image.new("RGB", OUTPUT_IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(image)

    for i in range(7):
        if i < 6:
            draw.line([(0, i * GRID_TILE_SIZE), (image.width, i * GRID_TILE_SIZE)], fill=GRID_BORDER_COLOR)
        else:
            draw.line([(0, i * GRID_TILE_SIZE - 1), (image.width, i * GRID_TILE_SIZE - 1)], fill=GRID_BORDER_COLOR)

    for i in range(7):
        if i < 6:
            draw.line([(i * GRID_TILE_SIZE, 0), (i * GRID_TILE_SIZE, image.height)], fill=GRID_BORDER_COLOR)
        else:
            draw.line([(i * GRID_TILE_SIZE - 1, 0), (i * GRID_TILE_SIZE - 1, image.height)], fill=GRID_BORDER_COLOR)

    for i in range(puzzle.rows_count()):
        for j in range(puzzle.cols_count()):
            symbol = puzzle.peek(i, j)

            if not symbol:
                continue

            symbol_image = PIECE_TO_IMAGE[symbol]

            image.paste(symbol_image, (i * GRID_TILE_SIZE + GRID_TILE_SIZE // 2 - symbol_image.width // 2, j * GRID_TILE_SIZE + GRID_TILE_SIZE // 2 - symbol_image.height // 2), symbol_image)

    return image

def main():
    builder = ProblemBuilder()
    puzzle = builder.build()

    image = generate_puzzle_image(puzzle)

    image.show()

if __name__ == "__main__":
    main()

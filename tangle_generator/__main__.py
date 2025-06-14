import multiprocessing
import time

from generator import (
    generate_puzzle_image,
    pretty_format_puzzle,
    worker,
)


def generate_with_processes(target_pieces, connections, num_processes=4):
    result_queue = multiprocessing.Queue()
    stop_event = multiprocessing.Event()

    processes = [
        multiprocessing.Process(
            target=worker, args=(target_pieces, connections, result_queue, stop_event)
        )
        for _ in range(num_processes)
    ]

    for p in processes:
        p.start()

    puzzle, iterations = result_queue.get()  # Wait for any one to finish
    stop_event.set()  # Signal all processes to stop

    for p in processes:
        p.join()

    return puzzle, iterations


def main():
    target_pieces = 12
    connections = 28
    num_processes = 12

    start = time.time()
    puzzle, iterations = generate_with_processes(
        target_pieces, connections, num_processes
    )
    end = time.time()

    image = generate_puzzle_image(puzzle)

    print("")
    print(f"Puzzle generated in {end - start:.2f} seconds")
    print(f"Iterations (winning process): {iterations}")
    print(f"Puzzle filled cells count: {puzzle.filled_cells_count()}")
    print("")
    print("Generated puzzle:")
    print(pretty_format_puzzle(puzzle))

    image.show()
    image.save(
        f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{puzzle.filled_cells_count()}_puzzle.png"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # For cross-platform compatibility
    main()

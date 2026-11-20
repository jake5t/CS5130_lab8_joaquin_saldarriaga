## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Utilities File

"""Helper classes and functions shared by every module of the package.

Process:
    * Exceptions with informative messages
    * Wall-clock timing helpers
    * Validation of grid, image-size and option arguments
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import statistics
from contextlib import contextmanager
from time import perf_counter
from typing import Any, Callable, Iterator

from . import config


## ********** Exceptions **********
## Base class of every error raised on purpose by this package
class MosaicError(ValueError):
    """Base class of the errors raised by the mosaic generator (a ValueError, so old callers keep working)."""


## Raised when an input image cannot be used
class InvalidImageError(MosaicError):
    """The input image is missing, empty, unreadable or has an unsupported shape."""


## Raised when the grid does not fit the image
class InvalidGridError(MosaicError):
    """The grid size is not positive, too large, or does not divide the image evenly."""


## Raised when no usable tile exists
class TileLoadError(MosaicError):
    """The tile folder is missing or contains no readable tile image."""


## ********** Timing **********
## Function 1 -> Measure the wall-clock time of a block of code
@contextmanager
def timer(store: dict[str, float], name: str) -> Iterator[None]:
    """Context manager that adds the elapsed seconds of a block to a dictionary.

    Args:
        * store (dict) -> Dictionary that receives the elapsed seconds under the given name
        * name (str) -> Key under which the time is accumulated

    Example:
        >>> timings = {}
        >>> with timer(timings, "work"):
        ...     _ = sum(range(1000))
        >>> "work" in timings
        True
    """
    ## Start the clock before the block runs
    start = perf_counter()

    ## Run the block and always record the time, even when it raises
    try:
        ## Run the block that is being timed
        yield

    ## Add the elapsed time to any previous measurement with the same name
    finally:
        ## Add the elapsed time to any previous measurement with the same name
        store[name] = store.get(name, 0.0) + (perf_counter() - start)


## Function 2 -> Repeat a call and summarize its timing
def time_callable(func: Callable[[], Any], repeats: int = 5, warmup: int = 1) -> dict[str, float]:
    """Call a function several times and summarize the wall-clock seconds of each call.

    Args:
        * func (Callable) -> Function without arguments to time
        * repeats (int) -> Number of timed calls (at least 1)
        * warmup (int) -> Number of untimed calls made first (caches, imports, first-call effects)

    Returns:
        * dict -> 'mean', 'std', 'min', 'max' and the list of 'runs' in seconds
    """
    ## Reject a meaningless number of repetitions
    if repeats < 1:
        ## Report the invalid number of repetitions
        raise MosaicError("repeats must be at least 1, got " + str(repeats))

    ## Untimed calls that absorb first-call effects
    for _ in range(warmup):
        ## Untimed warm-up call
        func()

    ## Timed calls, one measurement per call
    runs = []

    ## One timed call per repetition
    for _ in range(repeats):
        ## Start the clock
        start = perf_counter()

        ## Run the function once
        func()

        ## Store the elapsed seconds of this call
        runs.append(perf_counter() - start)

    ## Summary statistics of the runs
    return {
        "mean": statistics.fmean(runs),
        "std": statistics.stdev(runs) if len(runs) > 1 else 0.0,
        "min": min(runs),
        "max": max(runs),
        "runs": runs,
    }


## Function 3 -> Write a duration in a readable unit
def format_seconds(seconds: float) -> str:
    """Write a duration in seconds, milliseconds or microseconds.

    Args:
        * seconds (float) -> Duration in seconds

    Returns:
        * str -> Text such as '1.23 s', '4.56 ms' or '7.89 us'

    Example:
        >>> format_seconds(0.0123)
        '12.30 ms'
    """
    ## Choose the unit that keeps the number readable
    if seconds >= 1:
        ## Seconds are written with two decimals
        return f"{seconds:.2f} s"

    ## Durations between one millisecond and one second use milliseconds
    if seconds >= 1e-3:
        ## Write the duration in milliseconds
        return f"{seconds * 1e3:.2f} ms"

    ## Anything shorter is written in microseconds
    return f"{seconds * 1e6:.2f} us"


## ********** Validation **********
## Function 4 -> Turn a grid argument into (rows, columns)
def normalize_grid_size(grid_size: int | tuple[int, int]) -> tuple[int, int]:
    """Convert a grid size given as an int or as (rows, columns) into a validated pair.

    Args:
        * grid_size (int | tuple) -> Cells per side (square grid) or (rows, columns)

    Returns:
        * tuple -> Validated (rows, columns)

    Raises:
        * InvalidGridError -> If a value is not a positive integer or exceeds the configured maximum
    """
    ## A single integer means a square grid
    if isinstance(grid_size, int) and not isinstance(grid_size, bool):
        ## A square grid has as many rows as columns
        rows = columns = grid_size

    ## A pair means rows and columns
    elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
        ## A pair is unpacked into rows and columns
        rows, columns = grid_size

    ## Anything else cannot describe a grid
    else:
        ## Report the unsupported type of the grid argument
        raise InvalidGridError("grid_size must be an int or a (rows, columns) pair, got " + repr(grid_size))

    ## Both values must be positive integers within the limit
    for value in (rows, columns):
        ## Reject values that are not positive integers
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            ## Explain which grid was rejected
            raise InvalidGridError("Grid sizes must be positive integers, got " + repr(grid_size))

        ## Reject grids larger than the configured maximum
        if value > config.MAX_GRID_SIZE:
            ## Explain the limit that was exceeded
            raise InvalidGridError(f"Grid size {value} is larger than the maximum of {config.MAX_GRID_SIZE}")

    ## Return plain Python integers
    return int(rows), int(columns)


## Function 5 -> Turn an image-size argument into (width, height)
def normalize_image_size(image_size: int | tuple[int, int]) -> tuple[int, int]:
    """Convert an image size given as an int or as (width, height) into a validated pair.

    Args:
        * image_size (int | tuple) -> Side of a square image or (width, height)

    Returns:
        * tuple -> Validated (width, height) in pixels

    Raises:
        * InvalidImageError -> If a value is not an integer inside the allowed range
    """
    ## A single integer means a square image
    if isinstance(image_size, int) and not isinstance(image_size, bool):
        ## A square image has the same width and height
        width = height = image_size

    ## A pair means width and height
    elif isinstance(image_size, (tuple, list)) and len(image_size) == 2:
        ## A pair is unpacked into width and height
        width, height = image_size

    ## Anything else cannot describe an image size
    else:
        ## Report the unsupported type of the size argument
        raise InvalidImageError("image_size must be an int or a (width, height) pair, got " + repr(image_size))

    ## Both values must be integers inside the supported range
    for value in (width, height):
        ## Reject values that are not integers
        if not isinstance(value, int) or isinstance(value, bool):
            ## Explain which size was rejected
            raise InvalidImageError("Image sizes must be integers, got " + repr(image_size))

        ## Reject sides outside the supported range
        if not config.MIN_IMAGE_SIZE <= value <= config.MAX_IMAGE_SIZE:
            ## Explain the supported range
            raise InvalidImageError(
                f"Image side {value} is outside the supported range {config.MIN_IMAGE_SIZE}-{config.MAX_IMAGE_SIZE}"
            )

    ## Return plain Python integers
    return int(width), int(height)


## Function 6 -> Check that a grid divides an image evenly and return the cell size
def cell_dimensions(image_size: tuple[int, int], grid: tuple[int, int]) -> tuple[int, int]:
    """Return the (height, width) of one cell after checking that the grid fits the image.

    Args:
        * image_size (tuple) -> Image (width, height) in pixels
        * grid (tuple) -> Grid (rows, columns)

    Returns:
        * tuple -> Cell (height, width) in pixels

    Raises:
        * InvalidGridError -> If the grid does not divide the image evenly or the cells would be too small
    """
    ## Unpack the dimensions
    width, height = image_size

    ## Unpack the grid into rows and columns
    rows, columns = grid

    ## The reshape trick used for the grid needs exact multiples
    if height % rows or width % columns:
        ## Explain which grid does not fit the image
        raise InvalidGridError(
            f"A {rows}x{columns} grid does not divide a {width}x{height} image evenly; "
            f"choose a grid size that divides both dimensions"
        )

    ## Cells smaller than the minimum cannot hold a tile
    cell_height, cell_width = height // rows, width // columns

    ## Reject cells smaller than the minimum
    if min(cell_height, cell_width) < config.MIN_CELL_SIZE:
        ## Explain which cell size is too small
        raise InvalidGridError(f"Cells of {cell_width}x{cell_height} pixels are too small")

    ## Return the cell height and width
    return cell_height, cell_width


## Function 7 -> Validate a string option
def check_option(value: str, allowed: tuple[str, ...], label: str) -> str:
    """Return the value in lower case if it is one of the allowed options.

    Args:
        * value (str) -> Option chosen by the caller
        * allowed (tuple) -> Valid options
        * label (str) -> Name of the option for the error message

    Returns:
        * str -> The validated option

    Raises:
        * MosaicError -> If the value is not allowed
    """
    ## Accept the option in any capitalization
    if isinstance(value, str) and value.lower() in allowed:
        ## Return the option in lower case
        return value.lower()

    ## Otherwise explain which options exist
    raise MosaicError(f"Unknown {label} '{value}'. Choose one of: {', '.join(allowed)}")

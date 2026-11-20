## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Configuration File

"""Central place for every constant used by the mosaic generator package.

Process:
    * Default grid, image and tile dimensions
    * Supported options (grid sizes, tile families, matching modes)
    * Limits used for input validation
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


## ********** Default Processing Parameters **********
## Side (in pixels) of the square image every input is resized to
DEFAULT_IMAGE_SIZE: int = 512

## Number of cells across and down the image
DEFAULT_GRID_SIZE: int = 32

## Grid sizes offered by the user interface (as in Lab 1)
SUPPORTED_GRID_SIZES: tuple[int, ...] = (16, 32, 64)

## Image sizes used by the profiling and benchmarking suite
BENCHMARK_IMAGE_SIZES: tuple[int, ...] = (256, 512, 1024)


## ********** Tiles **********
## Procedural tile families that need no files on disk (as in Lab 1)
PROCEDURAL_TILE_SETS: tuple[str, ...] = ("Solid", "Diagonal", "Checkerboard")

## Tile family used when the caller does not choose one
DEFAULT_TILE_SET: str = "Diagonal"

## Folder with the image tiles that ship with the project
DEFAULT_TILE_DIRECTORY: Path = Path(__file__).resolve().parent.parent / "tiles"

## File extensions accepted when loading tiles from a folder
SUPPORTED_TILE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

## Base colors of the four procedural patterns of every family (RGB)
PROCEDURAL_TILE_COLORS: np.ndarray = np.array(
    [[220, 70, 55], [45, 125, 210], [45, 165, 105], [235, 180, 45]], dtype = np.float32
)

## Maximum number of cell sizes whose resized tiles are kept in memory
TILE_CACHE_MAX_ENTRIES: int = 8


## ********** Matching **********
## Feature used to compare a cell with a tile: 'mean' -> average RGB, 'quadrant' -> average RGB of each 2 x 2 sub-block
MATCH_MODES: tuple[str, ...] = ("mean", "quadrant")

## Search strategy: 'kdtree' -> scipy cKDTree, 'brute' -> vectorized distances to every tile, 'auto' -> pick by tile count
MATCH_ALGORITHMS: tuple[str, ...] = ("auto", "kdtree", "brute")

## Default matching feature and search strategy
DEFAULT_MATCH_MODE: str = "mean"

## Default search strategy
DEFAULT_MATCH_ALGORITHM: str = "auto"

## 'auto' uses the k-d tree from this many tiles on (measured: brute force wins below, the tree wins above)
KDTREE_MIN_TILES_LOW_DIM: int = 128

## Same threshold for high-dimensional features (12 numbers per tile), where trees lose their advantage
KDTREE_MIN_TILES_HIGH_DIM: int = 4096


## ********** Image Handling **********
## Resampling filters selectable for the (single) resize of the input image
RESAMPLING_FILTERS: dict[str, Image.Resampling] = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "box": Image.Resampling.BOX,
}

## Resampling filter used by default (the one used in Lab 1)
DEFAULT_RESAMPLING: str = "lanczos"


## ********** Validation Limits **********
## Smallest accepted side of the processed image
MIN_IMAGE_SIZE: int = 8

## Largest accepted side of the processed image (protects memory)
MAX_IMAGE_SIZE: int = 4096

## Largest accepted number of cells along one side of the grid
MAX_GRID_SIZE: int = 256

## Smallest accepted side of one cell, in pixels
MIN_CELL_SIZE: int = 1

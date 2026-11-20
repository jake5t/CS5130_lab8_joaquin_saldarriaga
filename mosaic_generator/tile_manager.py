## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Tile Manager File

"""Tile loading, caching and feature extraction.

Process:
    * Load the tiles once (from a folder of images or from a procedural family)
    * Resize the tiles to the cell size only once per cell size and keep them in memory
    * Pre-compute tile features (mean color, quadrant colors) and a k-d tree over them
    * Answer nearest-tile queries for all cells at once
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.spatial import cKDTree

from . import config
from .utils import MosaicError, TileLoadError, check_option


## ********** Logger of the Module **********
logger = logging.getLogger(__name__)


## Function 1 -> Create the procedural tiles of one family
def procedural_tiles(tile_set_name: str, cell_height: int, cell_width: int) -> np.ndarray:
    """Create the four procedural RGB tiles of a family (same patterns as Lab 1).

    Args:
        * tile_set_name (str) -> 'Solid', 'Diagonal' or 'Checkerboard'
        * cell_height (int) -> Height of every tile in pixels
        * cell_width (int) -> Width of every tile in pixels

    Returns:
        * np.ndarray -> uint8 stack with shape (4, cell_height, cell_width, 3)

    Raises:
        * MosaicError -> If the family is unknown
    """
    ## Normalized coordinates: x grows along the columns, y grows along the rows
    x, y = np.meshgrid(
        np.linspace(0, 1, cell_width, dtype = np.float32),
        np.linspace(0, 1, cell_height, dtype = np.float32),
    )

    ## The 'Solid' family: a constant tile, two ramps and their average
    if tile_set_name == "Solid":
        ## Patterns of the 'Solid' family
        patterns = [np.ones_like(x), x, y, (x + y) / 2]

    ## The 'Diagonal' family: ramps and diagonal structures
    elif tile_set_name == "Diagonal":
        ## Patterns of the 'Diagonal' family
        patterns = [x, y, np.abs(x - y), (x + y) / 2]

    ## The 'Checkerboard' family: 8 x 8 checks and their modulated versions
    elif tile_set_name == "Checkerboard":
        ## Black and white checks
        checks = ((np.floor(x * 8) + np.floor(y * 8)) % 2).astype(np.float32)

        ## Patterns of the 'Checkerboard' family
        patterns = [checks, 1 - checks, x * checks, y * (1 - checks)]

    ## Any other name is an error
    else:
        ## Report the unknown family
        raise MosaicError(f"Unknown tile set '{tile_set_name}'. Choose one of: {', '.join(config.PROCEDURAL_TILE_SETS)}")

    ## Color every pattern with its base color and convert to uint8
    colored = [np.clip(pattern[..., None] * color, 0, 255).astype(np.uint8) for pattern, color in zip(patterns, config.PROCEDURAL_TILE_COLORS)]

    ## Return the stack of colored tiles
    return np.stack(colored)


## Function 2 -> Read the tile images of a folder
def load_tile_images(tile_directory: str | Path) -> tuple[list[Image.Image], list[str], list[tuple[str, str]]]:
    """Read every readable image of a folder once, skipping (and reporting) the unreadable ones.

    Args:
        * tile_directory (str | Path) -> Folder with the tile images

    Returns:
        * tuple -> Loaded RGB images, their file names and a list of (file name, reason) for skipped files

    Raises:
        * TileLoadError -> If the folder does not exist or has no readable image
    """
    ## The folder must exist
    directory = Path(tile_directory)

    ## Report the missing folder
    if not directory.is_dir():
        ## Explain which folder does not exist
        raise TileLoadError(f"Tile folder '{directory}' does not exist")

    ## Containers for the results
    images: list[Image.Image] = []

    ## Names of the loaded tiles
    names: list[str] = []

    ## Files that could not be read, with the reason
    skipped: list[tuple[str, str]] = []

    ## One pass over the folder at startup (the only loop over files)
    for path in sorted(directory.iterdir()):

        ## Files that are not images by extension are ignored without a warning
        if path.suffix.lower() not in config.SUPPORTED_TILE_EXTENSIONS:
            ## Skip files that are not images
            continue

        ## Read and decode the file now, so a damaged file is detected here and not later
        try:
            ## Read and decode the file now
            with Image.open(path) as opened:
                ## Keep the RGB image
                images.append(opened.convert("RGB"))

                ## Keep the file name
                names.append(path.name)

        ## A damaged tile is skipped, not fatal
        except (UnidentifiedImageError, OSError, ValueError) as error:
            ## Record the file and the reason
            skipped.append((path.name, str(error)))

            ## Log the skipped file
            logger.warning("Skipping unreadable tile %s: %s", path.name, error)

    ## At least one tile is needed to build a mosaic
    if not images:
        ## Report that no tile could be read
        raise TileLoadError(f"No readable tile images were found in '{directory}'")

    ## Return the images, names and skipped files
    return images, names, skipped


## ********** Cached Data of One Cell Size **********
@dataclass
class TileBundle:
    """Tiles resized to one cell size together with their features and search trees.

    Attributes:
        * tiles (np.ndarray) -> uint8 tiles with shape (n_tiles, cell_height, cell_width, 3)
        * mean_features (np.ndarray) -> Mean RGB color of every tile, shape (n_tiles, 3)
        * quadrant_features (np.ndarray | None) -> Mean RGB of the 2 x 2 sub-blocks, shape (n_tiles, 12)
        * trees (dict) -> k-d trees already built, indexed by matching mode
    """

    tiles: np.ndarray
    mean_features: np.ndarray

    ## Quadrant features are built on demand
    quadrant_features: np.ndarray | None = None

    ## The k-d trees are built on demand, one per matching mode
    trees: dict[str, cKDTree] = field(default_factory = dict)


## Function 3 -> Compute the mean color of every tile
def mean_color_features(tiles: np.ndarray) -> np.ndarray:
    """Compute the average RGB color of every tile.

    Args:
        * tiles (np.ndarray) -> uint8 tiles with shape (n_tiles, H, W, 3)

    Returns:
        * np.ndarray -> float32 array with shape (n_tiles, 3)
    """
    ## One reduction over the two spatial axes
    return tiles.mean(axis = (1, 2), dtype = np.float32)


## Function 4 -> Compute the colors of the four quadrants of every tile
def quadrant_color_features(tiles: np.ndarray) -> np.ndarray:
    """Compute the average RGB color of the 2 x 2 sub-blocks of every tile (12 numbers per tile).

    Args:
        * tiles (np.ndarray) -> uint8 tiles with shape (n_tiles, H, W, 3) where H and W are even

    Returns:
        * np.ndarray -> float32 array with shape (n_tiles, 12)

    Raises:
        * MosaicError -> If the tile sides are odd
    """
    ## The quadrants only exist when both sides can be halved
    count, height, width = tiles.shape[:3]

    ## Reject a tile whose sides cannot be halved
    if height % 2 or width % 2:
        ## Explain the even-size requirement
        raise MosaicError(f"The 'quadrant' mode needs even cell sides, got {width}x{height} pixels")

    ## Split every tile into 2 x 2 blocks and average each block
    blocks = tiles.reshape(count, 2, height // 2, 2, width // 2, 3)

    ## Average each block and flatten to 12 numbers per tile
    return blocks.mean(axis = (2, 4), dtype = np.float32).reshape(count, 12)


## ********** Tile Manager **********
class TileManager:
    """Load tiles once and serve resized tiles, features and nearest-tile queries from caches.

    Args:
        * tile_directory (str | Path | None) -> Folder with tile images (use this OR tile_set)
        * tile_set (str | None) -> Procedural family 'Solid', 'Diagonal' or 'Checkerboard'
        * cache_size (int) -> Number of cell sizes kept in memory

    Example:
        >>> manager = TileManager(tile_set = "Diagonal")
        >>> manager.tiles_for(16, 16).shape
        (4, 16, 16, 3)
    """

    ## Function 5 -> Create the manager and load the tiles
    def __init__(
        self,
        tile_directory: str | Path | None = None,
        tile_set: str | None = None,
        cache_size: int = config.TILE_CACHE_MAX_ENTRIES,
    ) -> None:
        """Load the tiles of the chosen source and prepare empty caches.

        Args:
            * tile_directory (str | Path | None) -> Folder with tile images
            * tile_set (str | None) -> Procedural family name
            * cache_size (int) -> Number of cell sizes kept in memory (at least 1)

        Raises:
            * MosaicError -> If both sources are given, the family is unknown or cache_size < 1
            * TileLoadError -> If the folder is missing or has no readable tile
        """
        ## Exactly one source: a folder, or a procedural family (default)
        if tile_directory is not None and tile_set is not None:
            ## Report the contradictory sources
            raise MosaicError("Give either tile_directory or tile_set, not both")

        ## Reject an invalid cache size
        if cache_size < 1:
            ## Explain the minimum cache size
            raise MosaicError("cache_size must be at least 1")

        ## Procedural family (default when nothing is given)
        if tile_directory is None:
            ## Name of the procedural family
            name = tile_set if tile_set is not None else config.DEFAULT_TILE_SET

            ## Reject an unknown family
            if name not in config.PROCEDURAL_TILE_SETS:
                ## Explain which families exist
                raise MosaicError(f"Unknown tile set '{name}'. Choose one of: {', '.join(config.PROCEDURAL_TILE_SETS)}")

            ## Tiles come from a procedural family
            self.source = "procedural"

            ## Name of the tile source
            self.name = name

            ## Procedural tiles need no loaded images
            self._images: list[Image.Image] = []

            ## Names of the four procedural tiles
            self.tile_names: list[str] = [f"{name}-{i}" for i in range(4)]

            ## No file was skipped
            self.skipped: list[tuple[str, str]] = []

        ## Folder of images: read once, here
        else:
            ## Tiles come from a folder
            self.source = "directory"

            ## Name of the tile source
            self.name = str(tile_directory)

            ## Read the folder once, here
            self._images, self.tile_names, self.skipped = load_tile_images(tile_directory)

        ## Cache with least-recently-used eviction, protected for concurrent users (e.g. a web server)
        self._cache: OrderedDict[tuple[int, int], TileBundle] = OrderedDict()

        ## Maximum number of cell sizes kept in memory
        self._cache_size = cache_size

        ## Lock that protects the cache
        self._lock = threading.RLock()

        ## Counters that show how well the cache works
        self.cache_hits = 0

        ## Cache misses counter
        self.cache_misses = 0

    ## Function 6 -> Number of tiles
    @property
    def n_tiles(self) -> int:
        """Return the number of tiles of the set."""
        ## The procedural families always have four tiles
        return len(self.tile_names)

    ## Function 7 -> Resize the tiles of a folder to one cell size
    def _resize_images(self, cell_height: int, cell_width: int) -> np.ndarray:
        """Resize every loaded image to the cell size (area averaging to shrink, bicubic to enlarge).

        Args:
            * cell_height (int) -> Target tile height
            * cell_width (int) -> Target tile width

        Returns:
            * np.ndarray -> uint8 stack with shape (n_tiles, cell_height, cell_width, 3)
        """
        ## Allocate the output once
        stack = np.empty((len(self._images), cell_height, cell_width, 3), dtype = np.uint8)

        ## Resize each tile; images already at the target size are copied as they are
        for position, image in enumerate(self._images):

            ## Area averaging (BOX) is the right filter when shrinking, bicubic when enlarging
            if image.size == (cell_width, cell_height):
                ## Use the image as it is when it already has the cell size
                resized = image
            else:
                ## Shrinking when both sides get smaller or equal
                shrinking = image.width >= cell_width and image.height >= cell_height

                ## Area averaging to shrink, bicubic to enlarge
                resample = Image.Resampling.BOX if shrinking else Image.Resampling.BICUBIC

                ## Resize the tile to the cell size
                resized = image.resize((cell_width, cell_height), resample)

            ## Store the pixels
            stack[position] = np.asarray(resized)

        ## Return the stack of resized tiles
        return stack

    ## Function 8 -> Get (or build) everything cached for one cell size
    def bundle(self, cell_height: int, cell_width: int) -> TileBundle:
        """Return the tiles and mean features for a cell size, building them only the first time.

        Args:
            * cell_height (int) -> Cell height in pixels
            * cell_width (int) -> Cell width in pixels

        Returns:
            * TileBundle -> Cached tiles and features
        """
        ## The cell size is the cache key
        key = (cell_height, cell_width)

        ## The lock makes the check-and-build step atomic
        with self._lock:

            ## Cache hit: mark as recently used and return
            if key in self._cache:
                ## Mark the cell size as recently used
                self._cache.move_to_end(key)

                ## Count the cache hit
                self.cache_hits += 1

                ## Return the cached bundle
                return self._cache[key]

            ## Cache miss: build the tiles for this cell size
            self.cache_misses += 1

            ## Procedural tiles are generated for this cell size
            if self.source == "procedural":
                ## Generate the tiles
                tiles = procedural_tiles(self.name, cell_height, cell_width)
            else:
                ## Folder tiles are resized to this cell size
                tiles = self._resize_images(cell_height, cell_width)

            ## Pre-compute the mean-color features once and store everything
            bundle = TileBundle(tiles = tiles, mean_features = mean_color_features(tiles))

            ## Store the bundle in the cache
            self._cache[key] = bundle

            ## Evict the least recently used cell size when the cache is full
            while len(self._cache) > self._cache_size:
                ## Evict the least recently used cell size
                self._cache.popitem(last = False)

            ## Return the bundle
            return bundle

    ## Function 9 -> Resized tiles
    def tiles_for(self, cell_height: int, cell_width: int) -> np.ndarray:
        """Return the tiles resized to the cell size (cached).

        Args:
            * cell_height (int) -> Cell height in pixels
            * cell_width (int) -> Cell width in pixels

        Returns:
            * np.ndarray -> uint8 stack with shape (n_tiles, cell_height, cell_width, 3)
        """
        ## The bundle holds the tile stack
        return self.bundle(cell_height, cell_width).tiles

    ## Function 10 -> Features of the tiles
    def features_for(self, cell_height: int, cell_width: int, mode: str = config.DEFAULT_MATCH_MODE) -> np.ndarray:
        """Return the matching features of every tile for a cell size (cached).

        Args:
            * cell_height (int) -> Cell height in pixels
            * cell_width (int) -> Cell width in pixels
            * mode (str) -> 'mean' (3 numbers per tile) or 'quadrant' (12 numbers per tile)

        Returns:
            * np.ndarray -> float32 features with shape (n_tiles, 3) or (n_tiles, 12)
        """
        ## Validate the mode and fetch the cached bundle
        mode = check_option(mode, config.MATCH_MODES, "match mode")

        ## Cached bundle of this cell size
        bundle = self.bundle(cell_height, cell_width)

        ## Mean colors are always available
        if mode == "mean":
            ## Mean colors are always available
            return bundle.mean_features

        ## Quadrant features are built the first time they are requested
        with self._lock:
            ## Quadrant features are built the first time they are requested
            if bundle.quadrant_features is None:
                ## Build the quadrant features
                bundle.quadrant_features = quadrant_color_features(bundle.tiles)

        ## Return the quadrant features
        return bundle.quadrant_features

    ## Function 11 -> Nearest tile of every query
    def match(
        self,
        queries: np.ndarray,
        cell_height: int,
        cell_width: int,
        mode: str = config.DEFAULT_MATCH_MODE,
        algorithm: str = config.DEFAULT_MATCH_ALGORITHM,
    ) -> np.ndarray:
        """Find the tile closest (Euclidean distance) to every query feature vector.

        Args:
            * queries (np.ndarray) -> Cell features with shape (n_cells, D) where D is 3 or 12
            * cell_height (int) -> Cell height in pixels
            * cell_width (int) -> Cell width in pixels
            * mode (str) -> 'mean' or 'quadrant', must match D
            * algorithm (str) -> 'kdtree' (cKDTree), 'brute' (distances to every tile) or 'auto' (chosen by tile count)

        Returns:
            * np.ndarray -> int64 tile index of every query, shape (n_cells,)

        Raises:
            * MosaicError -> If the mode or algorithm is unknown or the feature size does not match
        """
        ## Validate the options and get the tile features
        algorithm = check_option(algorithm, config.MATCH_ALGORITHMS, "match algorithm")

        ## Validate the matching mode
        mode = check_option(mode, config.MATCH_MODES, "match mode")

        ## Tile features of this cell size
        features = self.features_for(cell_height, cell_width, mode)

        ## Queries and tile features must have the same number of columns
        if queries.shape[-1] != features.shape[1]:
            ## Report the mismatch between queries and tile features
            raise MosaicError(f"Query features have {queries.shape[-1]} values but the '{mode}' mode uses {features.shape[1]}")

        ## 'auto' picks the faster strategy for the size of the tile set (thresholds measured in the profiling notebook)
        if algorithm == "auto":
            ## Threshold of tiles from which the tree is faster
            threshold = config.KDTREE_MIN_TILES_LOW_DIM if features.shape[1] <= 3 else config.KDTREE_MIN_TILES_HIGH_DIM

            ## Choose the strategy
            algorithm = "kdtree" if features.shape[0] >= threshold else "brute"

        ## Strategy 1: one nearest-neighbor query on a tree built once per cell size
        if algorithm == "kdtree":
            ## Cached bundle of this cell size
            bundle = self.bundle(cell_height, cell_width)

            ## Build or fetch the tree under the lock
            with self._lock:
                ## Check whether the tree of this mode already exists
                if mode not in bundle.trees:
                    ## Build the tree the first time
                    bundle.trees[mode] = cKDTree(features)

                ## Fetch the tree of this mode
                tree = bundle.trees[mode]

            ## Nearest tile of every query
            return tree.query(queries, k = 1)[1].astype(np.int64)

        ## Strategy 2: squared distances to every tile through one matrix product, then arg-min per cell
        points = queries.astype(np.float64)

        ## Tile features as float64
        centers = features.astype(np.float64)

        ## Squared distances through one matrix product
        squared = (points ** 2).sum(axis = 1)[:, None] - 2.0 * points @ centers.T + (centers ** 2).sum(axis = 1)[None, :]

        ## Index of the nearest tile of every query
        return squared.argmin(axis = 1).astype(np.int64)

    ## Function 12 -> Forget everything cached
    def clear_cache(self) -> None:
        """Remove every cached tile stack, feature array and tree (the loaded images stay)."""
        ## Reset the cache and its counters
        with self._lock:
            ## Remove every cached bundle
            self._cache.clear()

            ## Reset the hit counter
            self.cache_hits = 0

            ## Reset the miss counter
            self.cache_misses = 0

    ## Function 13 -> Describe the manager
    def __repr__(self) -> str:
        """Return a short description of the tile source and the cache state."""
        ## Source, number of tiles and cached cell sizes
        return f"TileManager(source = {self.source!r}, name = {self.name!r}, n_tiles = {self.n_tiles}, cached_sizes = {len(self._cache)})"

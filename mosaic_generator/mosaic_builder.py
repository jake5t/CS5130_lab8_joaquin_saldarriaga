## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Mosaic Builder File

"""Main mosaic construction logic.

Process:
    * Prepare the image (one crop + resize)
    * Reduce every cell to its features with vectorized reductions
    * Find the nearest tile of all cells with one k-d tree query
    * Reassemble the chosen tiles into the mosaic and report per-stage timings
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from . import config
from .image_processor import assemble_mosaic, cell_means, prepare_image, segmented_image
from .metrics import compare_images
from .tile_manager import TileManager
from .utils import MosaicError, cell_dimensions, check_option, normalize_grid_size, normalize_image_size, timer


## ********** Result of One Mosaic **********
@dataclass
class MosaicResult:
    """Everything produced while building one mosaic.

    Attributes:
        * prepared (np.ndarray) -> Cropped and resized input image
        * mosaic (np.ndarray) -> Mosaic image built from tiles
        * tile_indices (np.ndarray) -> Tile index of every cell, shape (rows, columns)
        * segmented (np.ndarray | None) -> Block-segmented image (None when not requested)
        * timings (dict) -> Seconds of every stage and of the whole build
    """

    prepared: np.ndarray
    mosaic: np.ndarray
    tile_indices: np.ndarray

    ## Segmentation image, if it was requested
    segmented: np.ndarray | None = None

    ## Seconds of every stage
    timings: dict[str, float] = field(default_factory = dict)


## ********** Mosaic Builder **********
class MosaicBuilder:
    """Build photo mosaics from a TileManager using vectorized operations.

    Args:
        * tile_manager (TileManager) -> Source of tiles, features and nearest-tile search
        * grid_size (int | tuple) -> Cells per side or (rows, columns)
        * image_size (int | tuple) -> Side of the processed square image or (width, height)
        * match_mode (str) -> 'mean' (Lab 1 behaviour) or 'quadrant'
        * match_algorithm (str) -> 'auto' (default), 'kdtree' or 'brute'
        * resample (str) -> Filter of the single resize: 'lanczos', 'bicubic', 'bilinear' or 'box'

    Example:
        >>> builder = MosaicBuilder(TileManager(tile_set = "Diagonal"), grid_size = (32, 32))
        >>> builder.create_mosaic(np.zeros((300, 400, 3), dtype = np.uint8)).shape
        (512, 512, 3)
    """

    ## Function 1 -> Validate the configuration once
    def __init__(
        self,
        tile_manager: TileManager,
        grid_size: int | tuple[int, int] = config.DEFAULT_GRID_SIZE,
        image_size: int | tuple[int, int] = config.DEFAULT_IMAGE_SIZE,
        match_mode: str = config.DEFAULT_MATCH_MODE,
        match_algorithm: str = config.DEFAULT_MATCH_ALGORITHM,
        resample: str = config.DEFAULT_RESAMPLING,
    ) -> None:
        """Store and validate the configuration so that errors appear before any image is processed.

        Args:
            * tile_manager (TileManager) -> Source of tiles
            * grid_size (int | tuple) -> Grid cells
            * image_size (int | tuple) -> Processed image size
            * match_mode (str) -> Matching feature
            * match_algorithm (str) -> Search strategy
            * resample (str) -> Resampling filter

        Raises:
            * MosaicError -> If an option is unknown or the configuration is inconsistent
            * InvalidGridError -> If the grid does not divide the image evenly
        """
        ## The tile manager is required
        if not isinstance(tile_manager, TileManager):
            ## Report the wrong type of tile manager
            raise MosaicError("tile_manager must be a TileManager instance")

        ## Validate every option
        self.tile_manager = tile_manager

        ## Store the validated grid
        self._grid = normalize_grid_size(grid_size)

        ## Store the validated image size
        self._size = normalize_image_size(image_size)

        ## Store the validated matching mode
        self.match_mode = check_option(match_mode, config.MATCH_MODES, "match mode")

        ## Store the validated search strategy
        self.match_algorithm = check_option(match_algorithm, config.MATCH_ALGORITHMS, "match algorithm")

        ## Store the validated resampling filter
        self.resample = check_option(resample, tuple(config.RESAMPLING_FILTERS), "resampling filter")

        ## The grid must divide the image and the cell sides must be even for the quadrant mode
        self._cell = cell_dimensions(self._size, self._grid)

        ## The quadrant mode needs even cell sides
        if self.match_mode == "quadrant" and (self._cell[0] % 2 or self._cell[1] % 2):
            ## Explain the even-size requirement
            raise MosaicError(f"The 'quadrant' mode needs even cell sides, got {self._cell[1]}x{self._cell[0]} pixels")

        ## Timings of the last call to create_mosaic / build
        self.last_timings: dict[str, float] = {}

    ## Function 2 -> Read-only views of the configuration
    @property
    def grid_size(self) -> tuple[int, int]:
        """Return the grid as (rows, columns)."""
        ## Stored after validation
        return self._grid

    ## Read-only property: the grid
    @property
    def image_size(self) -> tuple[int, int]:
        """Return the processed image size as (width, height)."""
        ## Stored after validation
        return self._size

    ## Read-only property: the image size
    @property
    def cell_size(self) -> tuple[int, int]:
        """Return the cell size as (height, width) in pixels."""
        ## Derived from the image size and the grid
        return self._cell

    ## Function 3 -> Cell features for the chosen matching mode
    def _cell_features(self, prepared: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the mean color of every cell and the flat feature matrix used for matching.

        Args:
            * prepared (np.ndarray) -> Prepared RGB image

        Returns:
            * tuple -> Cell means with shape (rows, columns, 3) and query features with shape (rows * columns, D)
        """
        ## Mean mode: the query is the mean color of the cell
        rows, columns = self._grid

        ## Mean mode: queries are the mean colors
        if self.match_mode == "mean":
            ## Mean color of every cell
            means = cell_means(prepared, self._grid)

            ## Return the means and the flat query matrix
            return means, means.reshape(-1, 3)

        ## Quadrant mode: means of a grid twice as fine, regrouped into 12 numbers per cell
        fine = cell_means(prepared, (2 * rows, 2 * columns))

        ## Quadrant mode: 12 numbers per cell from a grid twice as fine
        queries = fine.reshape(rows, 2, columns, 2, 3).transpose(0, 2, 1, 3, 4).reshape(rows * columns, 12)

        ## The cell mean is the average of its four sub-block means
        means = fine.reshape(rows, 2, columns, 2, 3).mean(axis = (1, 3), dtype = np.float32)

        ## Return the means and the 12-number queries
        return means, queries

    ## Function 4 -> Build a mosaic and keep every intermediate result
    def build(self, image: Any, segmentation: bool = True) -> MosaicResult:
        """Build the mosaic of an image and return every intermediate result with stage timings.

        Args:
            * image (Any) -> File path, PIL image or NumPy array
            * segmentation (bool) -> Whether to also build the block-segmented image

        Returns:
            * MosaicResult -> Prepared image, mosaic, tile indices, optional segmentation and timings

        Raises:
            * InvalidImageError -> If the image cannot be used
        """
        ## Stage timings are collected in one dictionary
        timings: dict[str, float] = {}

        ## Grid size
        rows, columns = self._grid

        ## Cell size
        cell_height, cell_width = self._cell

        ## Stage 1: crop and resize once
        with timer(timings, "prepare"):
            ## Crop and resize the image
            prepared = prepare_image(image, self._size, self.resample)

        ## Stage 2: reduce every cell to its features
        with timer(timings, "grid"):
            ## Cell features
            means, queries = self._cell_features(prepared)

        ## Stage 3: nearest tile of all cells (tiles and features come from the cache)
        with timer(timings, "match"):
            ## Nearest tile of every cell
            indices = self.tile_manager.match(queries, cell_height, cell_width, self.match_mode, self.match_algorithm)

            ## Reshape the indices to the grid
            indices = indices.reshape(rows, columns)

        ## Stage 4: put the chosen tiles in place
        with timer(timings, "assemble"):
            ## Assemble the mosaic from the cached tiles
            mosaic = assemble_mosaic(self.tile_manager.tiles_for(cell_height, cell_width), indices)

        ## Optional stage: block-segmented picture of the cell colors
        segmented = None

        ## Build the segmentation if requested
        if segmentation:
            ## Time the segmentation
            with timer(timings, "segment"):
                ## Segmentation from the cell colors
                segmented = segmented_image(means, cell_height, cell_width)

        ## Total time of the stages
        timings["total"] = sum(timings.values())

        ## Remember the timings of this build
        self.last_timings = timings

        ## Return everything that was built
        return MosaicResult(prepared = prepared, mosaic = mosaic, tile_indices = indices, segmented = segmented, timings = timings)

    ## Function 5 -> The simple API of the assignment
    def create_mosaic(self, image: Any) -> np.ndarray:
        """Build the mosaic of an image (no segmentation and no metrics, which are optional extras).

        Args:
            * image (Any) -> File path, PIL image or NumPy array

        Returns:
            * np.ndarray -> RGB uint8 mosaic with shape (height, width, 3)
        """
        ## Only the mosaic is needed here
        return self.build(image, segmentation = False).mosaic

    ## Function 6 -> Batch processing
    def create_mosaics_batch(self, images: Iterable[Any]) -> list[np.ndarray]:
        """Build the mosaic of several images, sharing the tile caches.

        Args:
            * images (Iterable) -> Images as paths, PIL images or arrays

        Returns:
            * list -> One mosaic per image, in the same order
        """
        ## One call per image; the tile manager already holds every shared cache
        return [self.create_mosaic(image) for image in images]

    ## Function 7 -> Similarity between an image and its mosaic
    def compute_similarity(self, original: Any, mosaic: np.ndarray) -> dict[str, float]:
        """Compare an input image with its mosaic using MSE, PSNR and SSIM.

        Args:
            * original (Any) -> The input image (it is prepared like the mosaic input when its shape differs)
            * mosaic (np.ndarray) -> The mosaic returned by create_mosaic

        Returns:
            * dict -> 'MSE', 'PSNR' and 'SSIM'
        """
        ## An already prepared image is used as it is; anything else goes through the same preparation as the mosaic input
        if isinstance(original, np.ndarray) and original.shape == mosaic.shape and original.dtype == np.uint8:
            ## Use an already prepared image as it is
            reference = original
        else:
            ## Prepare any other input like the mosaic input
            reference = prepare_image(original, (mosaic.shape[1], mosaic.shape[0]), self.resample)

        ## Metrics against the mosaic
        return compare_images(reference, mosaic)

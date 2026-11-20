## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Image Processor File

"""Image loading, resizing and grid division with vectorized NumPy operations.

Process:
    * Load any supported input as an RGB uint8 array
    * Center-crop and resize the image in one pass
    * Split the image into grid cells and reduce every cell to its mean color
    * Expand cell colors (segmentation) and reassemble tiles (mosaic) without Python loops
"""

## ********** Import Required Libraries **********
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config
from .utils import InvalidImageError, cell_dimensions, check_option, normalize_grid_size, normalize_image_size


## Function 1 -> Convert any supported array into an RGB uint8 PIL image
def _array_to_pil(array: np.ndarray) -> Image.Image:
    """Convert a grayscale, RGB or RGBA array (uint8 or float) into an RGB PIL image.

    Args:
        * array (np.ndarray) -> Image array with shape (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)

    Returns:
        * Image.Image -> RGB PIL image

    Raises:
        * InvalidImageError -> If the array is empty or has an unsupported shape
    """
    ## An empty array has no pixels to process
    if array.size == 0:
        ## Report the empty image
        raise InvalidImageError("The input image is empty")

    ## Only 2-D (gray) and 3-D (channels) arrays describe an image
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] not in (1, 3, 4)):
        ## Report the unsupported array shape
        raise InvalidImageError(f"Unsupported image array of shape {array.shape}; expected (H, W), (H, W, 1), (H, W, 3) or (H, W, 4)")

    ## Float images are assumed to be in [0, 1] when their maximum is at most 1, otherwise in [0, 255]
    if array.dtype != np.uint8:
        ## Scale factor: [0, 1] floats become [0, 255]
        scale = 255.0 if np.nanmax(array) <= 1.0 else 1.0

        ## Clip, clean NaN values and convert to uint8
        array = np.clip(np.nan_to_num(array.astype(np.float64) * scale), 0, 255).astype(np.uint8)

    ## A single channel is stored as a plain 2-D array for Pillow
    if array.ndim == 3 and array.shape[2] == 1:
        ## Drop the channel axis of a single-channel image
        array = array[..., 0]

    ## Pillow handles the conversion of gray and RGBA inputs to RGB; an RGB array is wrapped without copying its pixels
    image = Image.fromarray(array)

    ## Return the image converted to RGB when needed
    return image if image.mode == "RGB" else image.convert("RGB")


## Function 2 -> Load an image from a path, a PIL image or an array
def load_image(source: Any) -> Image.Image:
    """Load an image from a file path, a PIL image or a NumPy array and return it in RGB.

    Args:
        * source (str | Path | Image.Image | np.ndarray) -> Where the image comes from

    Returns:
        * Image.Image -> RGB PIL image with the EXIF orientation applied (files only)

    Raises:
        * InvalidImageError -> If the file is missing or unreadable, or the input type is not supported

    Example:
        >>> load_image(np.zeros((8, 8, 3), dtype = np.uint8)).size
        (8, 8)
    """
    ## A missing input is the most common mistake in a user interface
    if source is None:
        ## Report the missing image
        raise InvalidImageError("No image was provided")

    ## Arrays are converted directly
    if isinstance(source, np.ndarray):
        ## Convert the array directly
        return _array_to_pil(source)

    ## PIL images only need the color conversion (an RGB image is used as it is, with no copy)
    if isinstance(source, Image.Image):
        ## Return the image converted to RGB when needed
        return source if source.mode == "RGB" else source.convert("RGB")

    ## Paths are opened with Pillow, which reports unreadable files
    if isinstance(source, (str, Path)):
        ## Open the file inside a guarded block
        try:
            ## Open the file with Pillow, which closes it afterwards
            with Image.open(source) as opened:
                ## Apply the EXIF orientation and convert to RGB
                return ImageOps.exif_transpose(opened).convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as error:
            ## Turn the low-level error into a clear message
            raise InvalidImageError(f"Could not read the image '{source}': {error}") from error

    ## Any other type cannot be interpreted as an image
    raise InvalidImageError("Unsupported image input of type " + type(source).__name__)


## Function 3 -> Compute the center-crop rectangle for a target aspect ratio
def center_crop_box(source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Return the centered crop rectangle with the aspect ratio of the target size.

    Args:
        * source_size (tuple) -> Source image (width, height)
        * target_size (tuple) -> Target image (width, height)

    Returns:
        * tuple -> Crop box (left, top, right, bottom) in source pixels
    """
    ## Aspect ratios of the target and of the source
    target_width, target_height = target_size

    ## Unpack the source and target sizes
    source_width, source_height = source_size

    ## Aspect ratio of the target
    target_ratio = target_width / target_height

    ## Aspect ratio of the source
    source_ratio = source_width / source_height

    ## A source that is too wide loses its left and right borders
    if source_ratio > target_ratio:
        ## Width of the centered crop
        crop_width = int(source_height * target_ratio)

        ## Left border that is removed
        left = (source_width - crop_width) // 2

        ## Return the crop box of a source that is too wide
        return left, 0, left + crop_width, source_height

    ## A source that is too tall loses its top and bottom borders
    crop_height = int(source_width / target_ratio)

    ## Top border that is removed
    top = (source_height - crop_height) // 2

    ## Return the crop box of a source that is too tall
    return 0, top, source_width, top + crop_height


## Function 4 -> Prepare an input image for grid processing
def prepare_image(
    image: Any,
    output_size: int | tuple[int, int] = config.DEFAULT_IMAGE_SIZE,
    resample: str = config.DEFAULT_RESAMPLING,
) -> np.ndarray:
    """Center-crop and resize an input image to an RGB uint8 array with a single resampling pass.

    Args:
        * image (Any) -> File path, PIL image or NumPy array
        * output_size (int | tuple) -> Side of the square output or its (width, height)
        * resample (str) -> Resampling filter: 'lanczos', 'bicubic', 'bilinear' or 'box'

    Returns:
        * np.ndarray -> RGB uint8 array with shape (height, width, 3)

    Raises:
        * InvalidImageError -> If the image or the size is not valid
        * MosaicError -> If the resampling filter is unknown
    """
    ## Validate the options before touching any pixel
    size = normalize_image_size(output_size)

    ## Validate the resampling filter name
    resample_name = check_option(resample, tuple(config.RESAMPLING_FILTERS), "resampling filter")

    ## Load the image as RGB
    pil_image = load_image(image)

    ## Crop is lazy in Pillow (no pixel copy); the single resize then reads only the cropped region
    crop_box = center_crop_box(pil_image.size, size)

    ## Crop and resize: the crop is lazy, the resize reads only the cropped region
    resized = pil_image.crop(crop_box).resize(size, config.RESAMPLING_FILTERS[resample_name])

    ## Return the pixels as a contiguous uint8 array
    return np.asarray(resized, dtype = np.uint8)


## Function 5 -> Split an image into its grid cells
def split_into_cells(image: np.ndarray, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Split an RGB image into grid cells with reshape and transpose (no Python loops).

    Args:
        * image (np.ndarray) -> RGB image with shape (H, W, 3) whose sides are multiples of the grid
        * grid_size (int | tuple) -> Cells per side or (rows, columns)

    Returns:
        * np.ndarray -> Array of shape (rows * columns, cell_height, cell_width, 3) in row-major cell order

    Example:
        >>> split_into_cells(np.zeros((8, 8, 3), dtype = np.uint8), 2).shape
        (4, 4, 4, 3)
    """
    ## Grid and cell dimensions, validated against the image
    rows, columns = normalize_grid_size(grid_size)

    ## Image height and width
    height, width = image.shape[:2]

    ## Cell size, validated against the grid
    cell_height, cell_width = cell_dimensions((width, height), (rows, columns))

    ## Reshape to (grid row, cell row, grid column, cell column, channel) and bring the grid axes to the front
    cells = image.reshape(rows, cell_height, columns, cell_width, 3).transpose(0, 2, 1, 3, 4)

    ## Flatten the two grid axes into one list of cells
    return cells.reshape(-1, cell_height, cell_width, 3)


## Function 6 -> Reduce every cell to its mean color
def cell_means(image: np.ndarray, grid_size: int | tuple[int, int]) -> np.ndarray:
    """Compute the mean RGB color of every grid cell with two contiguous integer reductions.

    Args:
        * image (np.ndarray) -> RGB uint8 image with shape (H, W, 3)
        * grid_size (int | tuple) -> Cells per side or (rows, columns)

    Returns:
        * np.ndarray -> float32 array with shape (rows, columns, 3)

    Example:
        >>> cell_means(np.full((8, 8, 3), 10, dtype = np.uint8), 2)[0, 0]
        array([10., 10., 10.], dtype=float32)
    """
    ## Grid and cell dimensions, validated against the image
    rows, columns = normalize_grid_size(grid_size)

    ## Image height and width
    height, width = image.shape[:2]

    ## Cell size, validated against the grid
    cell_height, cell_width = cell_dimensions((width, height), (rows, columns))

    ## Integer sums are exact; uint16 is enough while a column of cells sums to at most 65535
    row_dtype = np.uint16 if cell_height <= 257 else np.uint32

    ## Step 1: add the rows of every cell with one reduction over a contiguous axis
    row_sums = image.reshape(rows, cell_height, width * 3).sum(axis = 1, dtype = row_dtype)

    ## Step 2: add the columns of every cell and divide by the number of pixels
    cell_sums = row_sums.reshape(rows, columns, cell_width, 3).sum(axis = 2, dtype = np.uint32)

    ## Convert the sums to means in float32
    return cell_sums.astype(np.float32) / np.float32(cell_height * cell_width)


## Function 7 -> Expand cell colors into a block-segmented image
def segmented_image(means: np.ndarray, cell_height: int, cell_width: int) -> np.ndarray:
    """Expand one color per cell into an image where every cell has a uniform color.

    Args:
        * means (np.ndarray) -> Cell colors with shape (rows, columns, 3)
        * cell_height (int) -> Height of one cell in pixels
        * cell_width (int) -> Width of one cell in pixels

    Returns:
        * np.ndarray -> RGB uint8 image with shape (rows * cell_height, columns * cell_width, 3)
    """
    ## Colors are truncated to uint8 as in Lab 1
    rows, columns = means.shape[:2]

    ## Truncate the colors to uint8 as Lab 1 does
    colors = np.clip(means, 0, 255).astype(np.uint8)

    ## Broadcast every color over its cell and merge the axes (one copy, no repeat calls)
    blocks = np.broadcast_to(colors[:, None, :, None, :], (rows, cell_height, columns, cell_width, 3))

    ## Merge the axes and return the segmented image
    return blocks.reshape(rows * cell_height, columns * cell_width, 3)


## Function 8 -> Reassemble selected tiles into one image
def assemble_mosaic(tiles: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Place the chosen tile in every cell with fancy indexing and one transpose.

    Args:
        * tiles (np.ndarray) -> Tile stack with shape (n_tiles, cell_height, cell_width, 3)
        * indices (np.ndarray) -> Tile index of every cell, shape (rows, columns)

    Returns:
        * np.ndarray -> RGB uint8 mosaic with shape (rows * cell_height, columns * cell_width, 3)
    """
    ## Gather the tile of every cell: shape (rows, columns, cell_height, cell_width, 3)
    rows, columns = indices.shape

    ## Gather the tile of every cell
    chosen = tiles[indices]

    ## Height and width of one tile
    cell_height, cell_width = chosen.shape[2:4]

    ## Interleave grid rows with cell rows and grid columns with cell columns
    return chosen.transpose(0, 2, 1, 3, 4).reshape(rows * cell_height, columns * cell_width, 3)

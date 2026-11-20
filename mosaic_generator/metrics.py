## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Similarity Metrics File

"""Similarity metrics between an original image and its mosaic.

Metrics:
    * Mean Squared Error (MSE), where lower values mean a smaller pixel error
    * Peak Signal-to-Noise Ratio (PSNR), the same error on a logarithmic scale
    * Structural Similarity (SSIM), where values near one mean a stronger similarity
    * Tile agreement, the fraction of cells that received the same tile in two mosaics
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from .utils import MosaicError


## Function 1 -> Check that two images can be compared
def _check_same_shape(first: np.ndarray, second: np.ndarray) -> None:
    """Raise an error if two images do not have the same shape.

    Args:
        * first (np.ndarray) -> First image
        * second (np.ndarray) -> Second image

    Raises:
        * MosaicError -> If the shapes differ
    """
    ## Metrics are computed pixel by pixel, so the shapes must be equal
    if first.shape != second.shape:
        ## Report the shape mismatch
        raise MosaicError(f"Images must have the same shape to be compared, got {first.shape} and {second.shape}")


## Function 2 -> Mean Squared Error
def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the average squared pixel error between two images.

    Args:
        * original (np.ndarray) -> The original image
        * reconstructed (np.ndarray) -> The reconstructed image

    Returns:
        * float -> Mean squared pixel difference

    Example:
        >>> mean_squared_error(np.zeros((2, 2, 3), dtype = np.uint8), np.full((2, 2, 3), 2, dtype = np.uint8))
        4.0
    """
    ## Same shape is required
    _check_same_shape(original, reconstructed)

    ## Convert to float before subtracting to avoid uint8 wrap-around
    difference = original.astype(np.float32) - reconstructed.astype(np.float32)

    ## Average of the squared differences over all pixels and channels
    return float(np.mean(difference ** 2))


## Function 3 -> Peak Signal-to-Noise Ratio
def peak_signal_noise_ratio(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the PSNR in decibels (infinite when the images are identical).

    Args:
        * original (np.ndarray) -> The original image
        * reconstructed (np.ndarray) -> The reconstructed image

    Returns:
        * float -> 10 * log10(255^2 / MSE)
    """
    ## PSNR is a logarithmic rescaling of the MSE
    error = mean_squared_error(original, reconstructed)

    ## PSNR is infinite when the images are identical
    return float("inf") if error == 0 else float(10.0 * np.log10(255.0 ** 2 / error))


## Function 4 -> Structural Similarity
def structural_similarity_index(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Return the multichannel structural similarity between two RGB images (scikit-image).

    Args:
        * original (np.ndarray) -> The original RGB image
        * reconstructed (np.ndarray) -> The reconstructed RGB image

    Returns:
        * float -> SSIM score, 1.0 for identical images
    """
    ## Same shape is required
    _check_same_shape(original, reconstructed)

    ## The SSIM window must be odd and not larger than the smallest side
    smallest_side = min(original.shape[:2])

    ## Odd window size not larger than the smallest side
    window = min(7, smallest_side if smallest_side % 2 else smallest_side - 1)

    ## Reject images that are too small
    if window < 3:
        ## Explain the minimum size
        raise MosaicError("Images are too small to compute the SSIM")

    ## Compute the SSIM over the three channels
    return float(structural_similarity(original, reconstructed, channel_axis = -1, data_range = 255, win_size = window))


## Function 5 -> All metrics at once
def compare_images(original: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    """Return MSE, PSNR and SSIM of one reconstruction.

    Args:
        * original (np.ndarray) -> The original RGB image
        * reconstructed (np.ndarray) -> The reconstructed RGB image

    Returns:
        * dict -> 'MSE', 'PSNR' and 'SSIM' values
    """
    ## Compute and return every metric
    return {
        "MSE": mean_squared_error(original, reconstructed),
        "PSNR": peak_signal_noise_ratio(original, reconstructed),
        "SSIM": structural_similarity_index(original, reconstructed),
    }


## Function 6 -> Fraction of cells with the same tile
def tile_agreement(first_indices: np.ndarray, second_indices: np.ndarray) -> float:
    """Return the fraction of cells that were assigned the same tile in two mosaics.

    Args:
        * first_indices (np.ndarray) -> Tile index of every cell in the first mosaic
        * second_indices (np.ndarray) -> Tile index of every cell in the second mosaic

    Returns:
        * float -> Value in [0, 1]; 1.0 means identical assignments

    Example:
        >>> tile_agreement(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0]))
        0.75
    """
    ## Same shape is required
    _check_same_shape(np.asarray(first_indices), np.asarray(second_indices))

    ## Share of equal entries
    return float(np.mean(np.asarray(first_indices) == np.asarray(second_indices)))

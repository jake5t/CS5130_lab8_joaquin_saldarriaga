## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Unit Tests File

"""Unit tests of the mosaic generator package (run with 'pytest tests').

Process:
    * Image processing: loading, preparation, grid division, cell means
    * Tile management: caching, folder loading, nearest-tile search
    * Mosaic construction: identical output to the Lab 1 baseline and error handling
    * Metrics: MSE, PSNR, SSIM and tile agreement
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

## Make the package and the Lab 1 baseline importable when pytest runs from any folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

## Make the package importable from the tests
sys.path.insert(0, str(PROJECT_ROOT))

## Make the Lab 1 baseline importable from the tests
sys.path.insert(0, str(PROJECT_ROOT / "baseline_lab1"))

from mosaic_generator import InvalidGridError, InvalidImageError, MosaicBuilder, MosaicError, TileLoadError, TileManager
from mosaic_generator import image_processor, metrics
from mosaic_generator.utils import format_seconds, time_callable


## ********** Shared Test Data **********
## Random RGB image used by most tests (fixed seed -> reproducible)
@pytest.fixture(scope = "module")
def photo() -> np.ndarray:
    """Return a 300 x 400 random RGB uint8 image."""
    ## Fixed seed so that every run sees the same pixels
    return np.random.default_rng(42).integers(0, 256, size = (300, 400, 3), dtype = np.uint8)


## Function 1 -> Run the Lab 1 pipeline without its printed output
def lab1_mosaic(image: np.ndarray, grid_size: int, tile_set: str, image_size: int):
    """Call the Lab 1 create_mosaic function and silence its prints."""
    ## Import here so that a missing baseline folder only skips the tests that need it
    import mosaic_pipeline

    ## Hide the metric and benchmark prints of Lab 1
    with contextlib.redirect_stdout(io.StringIO()):
        ## Run the Lab 1 pipeline on the image
        return mosaic_pipeline.create_mosaic(image, grid_size, tile_set, image_size)


## ********** Image Processor **********
## Test 1 -> Every supported input becomes an RGB uint8 image
@pytest.mark.parametrize("shape", [(20, 30), (20, 30, 1), (20, 30, 3), (20, 30, 4)])
def test_prepare_image_accepts_gray_rgb_rgba(shape):
    """Grayscale, RGB and RGBA arrays are prepared to the requested size."""
    ## Build the array and prepare it
    array = np.random.default_rng(0).integers(0, 256, size = shape, dtype = np.uint8)

    ## Prepare a small image
    prepared = image_processor.prepare_image(array, 16)

    ## Output is always (16, 16, 3) uint8
    assert prepared.shape == (16, 16, 3)

    ## The prepared image must be uint8
    assert prepared.dtype == np.uint8


## Test 2 -> Float images in [0, 1] are scaled
def test_prepare_image_scales_float_images():
    """A float image in [0, 1] is converted to the 0-255 range."""
    ## All-ones float image must become all-255
    prepared = image_processor.prepare_image(np.ones((16, 16, 3), dtype = np.float32), 16)

    ## A float image of ones must become 255
    assert prepared.min() == 255


## Test 3 -> Same pixels as Lab 1 preprocessing
def test_prepare_image_matches_lab1(photo):
    """The crop + resize gives exactly the Lab 1 pixels."""
    ## Compare against the Lab 1 function
    from preprocessing import prepare_image as lab1_prepare

    ## Several sizes and aspect ratios
    for size in (64, 128):
        ## Lab 1 and the new function must give identical pixels
        assert np.array_equal(lab1_prepare(photo, (size, size)), image_processor.prepare_image(photo, size))


## Test 4 -> Errors of the image functions
def test_image_errors(tmp_path):
    """Unusable inputs raise InvalidImageError with a clear message."""
    ## Missing, empty, wrongly shaped, wrong type and unreadable inputs
    with pytest.raises(InvalidImageError):
        ## A missing image is rejected
        image_processor.prepare_image(None, 64)

    ## An empty array is rejected
    with pytest.raises(InvalidImageError):
        ## An empty array is rejected
        image_processor.prepare_image(np.zeros((0, 5, 3), dtype = np.uint8), 64)

    ## A two-channel array is rejected
    with pytest.raises(InvalidImageError):
        ## A two-channel array is rejected
        image_processor.prepare_image(np.zeros((5, 5, 2), dtype = np.uint8), 64)

    ## A number is not an image
    with pytest.raises(InvalidImageError):
        ## A number is not an image
        image_processor.prepare_image(12345, 64)

    ## A missing file is rejected
    with pytest.raises(InvalidImageError):
        ## A missing file is rejected
        image_processor.prepare_image(tmp_path / "missing.png", 64)

    ## A text file with an image extension
    fake = tmp_path / "fake.png"

    ## Write a file that is not an image
    fake.write_text("not an image")

    ## The file cannot be decoded
    with pytest.raises(InvalidImageError):
        ## The file cannot be decoded
        image_processor.prepare_image(fake, 64)

    ## An absurd output size
    with pytest.raises(InvalidImageError):
        ## An absurdly large size is rejected
        image_processor.prepare_image(np.zeros((8, 8, 3), dtype = np.uint8), 100000)


## Test 5 -> Unknown resampling filter
def test_unknown_resampling_filter():
    """An unknown resampling filter is rejected."""
    ## The message lists the valid filters
    with pytest.raises(MosaicError, match = "lanczos"):
        ## An unknown resampling filter is rejected
        image_processor.prepare_image(np.zeros((8, 8, 3), dtype = np.uint8), 8, resample = "nearest-ish")


## Test 6 -> Cell means equal the straightforward loop
def test_cell_means_match_loop(photo):
    """The two-step integer reduction equals the mean of every cell computed one by one."""
    ## Prepared image and its grid
    image = image_processor.prepare_image(photo, 128)

    ## Vectorized cell means
    means = image_processor.cell_means(image, 8)

    ## Reference computed with explicit loops
    expected = np.empty((8, 8, 3), dtype = np.float32)

    ## Reference means computed with loops
    for row in range(8):
        ## Loop over the columns
        for column in range(8):
            ## Mean of the cell, computed directly
            expected[row, column] = image[row * 16:(row + 1) * 16, column * 16:(column + 1) * 16].mean(axis = (0, 1))

    ## Equal up to float32 rounding
    assert np.allclose(means, expected, atol = 1e-4)


## Test 7 -> Splitting and reassembling cells gives back the image
def test_split_into_cells_roundtrip(photo):
    """Cells in row-major order can be put back into the original image."""
    ## Split an image into 4 x 8 cells
    image = image_processor.prepare_image(photo, (64, 32))

    ## Split the image into 4 x 8 cells
    cells = image_processor.split_into_cells(image, (4, 8))

    ## There must be 32 cells of 8 x 8 pixels
    assert cells.shape == (32, 8, 8, 3)

    ## Reassemble with the same transpose used for mosaics
    rebuilt = cells.reshape(4, 8, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(32, 64, 3)

    ## Reassembling the cells must give the original image
    assert np.array_equal(rebuilt, image)


## Test 8 -> Segmented image has uniform cells
def test_segmented_image_is_uniform(photo):
    """Every cell of the segmented image has a single color."""
    ## Segment a prepared image
    image = image_processor.prepare_image(photo, 64)

    ## Mean color of every cell
    means = image_processor.cell_means(image, 4)

    ## Segmented image from the means
    segmented = image_processor.segmented_image(means, 16, 16)

    ## All pixels of the first cell are equal
    assert segmented.shape == (64, 64, 3)

    ## The first cell must be one flat color
    assert (segmented[:16, :16] == segmented[0, 0]).all()


## ********** Tile Manager **********
## Test 9 -> Procedural tiles are generated once per cell size
def test_tile_manager_caches_cell_sizes():
    """The second request for a cell size is a cache hit and returns the same object."""
    ## Two requests with the same size, one with another size
    manager = TileManager(tile_set = "Diagonal")

    ## First request: a cache miss
    first = manager.tiles_for(16, 16)

    ## Second request: a cache hit
    second = manager.tiles_for(16, 16)

    ## Another cell size: another miss
    manager.tiles_for(8, 8)

    ## One hit, two misses, same array
    assert first is second

    ## Counters of hits and misses
    assert manager.cache_hits == 1 and manager.cache_misses == 2


## Test 10 -> Least-recently-used eviction
def test_tile_manager_evicts_old_sizes():
    """When the cache is full, the least recently used cell size is dropped."""
    ## A cache of two entries filled with three sizes
    manager = TileManager(tile_set = "Solid", cache_size = 2)

    ## Request three cell sizes
    for side in (4, 8, 16):
        ## Each size is cached separately
        manager.tiles_for(side, side)

    ## The oldest size must be rebuilt (a new miss)
    misses_before = manager.cache_misses

    ## Request the oldest size again
    manager.tiles_for(4, 4)

    ## It had been evicted, so it is a new miss
    assert manager.cache_misses == misses_before + 1


## Test 11 -> Loading tiles from a folder
def test_tile_manager_from_directory():
    """The shipped tile folder loads and resizes to any cell size."""
    ## Load the library once
    manager = TileManager(tile_directory = PROJECT_ROOT / "tiles")

    ## Resized tiles and features have consistent shapes
    assert manager.n_tiles == 216

    ## Shape of the stack of tiles
    assert manager.tiles_for(16, 16).shape == (216, 16, 16, 3)

    ## Shape of the mean features
    assert manager.features_for(16, 16, "mean").shape == (216, 3)

    ## Shape of the quadrant features
    assert manager.features_for(16, 16, "quadrant").shape == (216, 12)


## Test 12 -> Damaged tiles are skipped, missing folders are reported
def test_tile_manager_error_handling(tmp_path):
    """Unreadable tiles are skipped with a record; empty or missing folders raise TileLoadError."""
    ## A folder with one good tile and one damaged tile
    Image.fromarray(np.full((8, 8, 3), 100, dtype = np.uint8)).save(tmp_path / "good.png")

    ## Write a corrupted file
    (tmp_path / "bad.png").write_bytes(b"this is not a png")

    ## Load the folder
    manager = TileManager(tile_directory = tmp_path)

    ## The good tile is used, the bad one is reported
    assert manager.n_tiles == 1

    ## The corrupted file is reported as skipped
    assert manager.skipped and manager.skipped[0][0] == "bad.png"

    ## A folder without readable images
    empty = tmp_path / "empty"

    ## Create an empty folder
    empty.mkdir()

    ## Put only a corrupted file in it
    (empty / "bad.png").write_bytes(b"nope")

    ## A folder without readable tiles is rejected
    with pytest.raises(TileLoadError):
        ## A folder without readable tiles is rejected
        TileManager(tile_directory = empty)

    ## A folder that does not exist
    with pytest.raises(TileLoadError):
        ## A missing folder is rejected
        TileManager(tile_directory = tmp_path / "missing")


## Test 13 -> Invalid tile manager arguments
def test_tile_manager_argument_errors():
    """Contradictory or unknown arguments raise MosaicError."""
    ## Both sources, unknown family, invalid cache size
    with pytest.raises(MosaicError):
        ## Two sources at once are rejected
        TileManager(tile_directory = "tiles", tile_set = "Solid")

    ## An unknown family is rejected
    with pytest.raises(MosaicError):
        ## An unknown family is rejected
        TileManager(tile_set = "Plaid")

    ## A cache of size zero is rejected
    with pytest.raises(MosaicError):
        ## A cache of size zero is rejected
        TileManager(tile_set = "Solid", cache_size = 0)


## Test 14 -> k-d tree and brute force agree
@pytest.mark.parametrize("mode", ["mean", "quadrant"])
def test_kdtree_equals_brute_force(mode):
    """Both search strategies return the same nearest tile for random queries."""
    ## 216 file tiles and random queries of the right dimension
    manager = TileManager(tile_directory = PROJECT_ROOT / "tiles")

    ## Feature dimension of the mode
    dimension = 3 if mode == "mean" else 12

    ## Random queries
    queries = np.random.default_rng(1).uniform(0, 255, size = (500, dimension)).astype(np.float32)

    ## Same answers
    tree = manager.match(queries, 16, 16, mode, "kdtree")

    ## Brute-force result
    brute = manager.match(queries, 16, 16, mode, "brute")

    ## Both strategies must agree
    assert np.array_equal(tree, brute)


## Test 14b -> 'auto' picks the same answer as the explicit strategies
@pytest.mark.parametrize("directory", [None, PROJECT_ROOT / "tiles"])
def test_auto_matches_explicit_strategies(directory):
    """The 'auto' strategy returns the same tiles as brute force (small set) or the k-d tree (large set)."""
    ## 4 procedural tiles or 216 file tiles
    manager = TileManager(tile_directory = directory) if directory else TileManager(tile_set = "Diagonal")

    ## Random queries
    queries = np.random.default_rng(2).uniform(0, 255, size = (300, 3)).astype(np.float32)

    ## Same answers as the explicit strategies
    auto = manager.match(queries, 16, 16, "mean", "auto")

    ## Automatic result
    assert np.array_equal(auto, manager.match(queries, 16, 16, "mean", "brute"))

    ## It must equal the brute-force result
    assert np.array_equal(auto, manager.match(queries, 16, 16, "mean", "kdtree"))


## Test 15 -> Wrong query size
def test_match_rejects_wrong_feature_size():
    """Queries with the wrong number of features raise MosaicError."""
    ## Quadrant tiles need 12 numbers per query
    manager = TileManager(tile_set = "Diagonal")

    ## It must equal the k-d tree result
    with pytest.raises(MosaicError):
        ## Queries with the wrong dimension are rejected
        manager.match(np.zeros((5, 3), dtype = np.float32), 16, 16, "quadrant")


## ********** Mosaic Builder **********
## Test 16 -> Identical to Lab 1 for every tile family and several sizes
@pytest.mark.parametrize("tile_set", ["Solid", "Diagonal", "Checkerboard"])
@pytest.mark.parametrize("image_size,grid", [(64, 4), (128, 16), (256, 32)])
def test_builder_matches_lab1(photo, tile_set, image_size, grid):
    """Prepared image, segmentation and mosaic are identical to the Lab 1 outputs."""
    ## Lab 1 and the optimized builder on the same input
    prepared, segmented, mosaic = lab1_mosaic(photo, grid, tile_set, image_size)

    ## Queries with the wrong dimension are rejected
    result = MosaicBuilder(TileManager(tile_set = tile_set), grid_size = grid, image_size = image_size).build(photo)

    ## Pixel-exact equality
    assert np.array_equal(prepared, result.prepared)

    ## Build the mosaic with the new builder
    assert np.array_equal(segmented, result.segmented)

    ## The segmentation must match Lab 1
    assert np.array_equal(mosaic, result.mosaic)


## Test 17 -> Exact tiles are recovered
def test_builder_recovers_a_known_tile_layout():
    """An image made of tiles is rebuilt with the same tile in every cell."""
    ## Random tile layout assembled from the procedural tiles
    manager = TileManager(tile_set = "Diagonal")

    ## The mosaic must match Lab 1
    layout = np.random.default_rng(3).integers(0, 4, size = (8, 8))

    ## Random layout of tiles
    image = image_processor.assemble_mosaic(manager.tiles_for(8, 8), layout)

    ## The builder must pick the same tile in every cell
    result = MosaicBuilder(manager, grid_size = 8, image_size = 64).build(image)

    ## Assemble the layout
    assert np.array_equal(result.tile_indices, layout)

    ## The layout is returned unchanged
    assert np.array_equal(result.mosaic, image)


## Test 18 -> Rectangular grids and images
def test_builder_rectangular_grid(photo):
    """A (rows, columns) grid on a non-square image produces the expected shapes."""
    ## 4 x 8 grid on a 64 x 32 image (width x height)
    builder = MosaicBuilder(TileManager(tile_set = "Checkerboard"), grid_size = (4, 8), image_size = (64, 32))

    ## The assembled image must match
    result = builder.build(photo)

    ## Cells are 8 x 8 pixels
    assert builder.cell_size == (8, 8)

    ## Build a mosaic from the photo
    assert result.mosaic.shape == (32, 64, 3)

    ## Shape of the mosaic
    assert result.tile_indices.shape == (4, 8)


## Test 19 -> The assignment's example usage
def test_example_usage(photo):
    """The API shown in the assignment text works as written."""
    ## Folder tiles, tuple grid, mosaic and similarity
    tile_manager = TileManager(tile_directory = PROJECT_ROOT / "tiles")

    ## Shape of the tile indices
    builder = MosaicBuilder(tile_manager, grid_size = (32, 32))

    ## Builder for a 32 x 32 grid
    mosaic = builder.create_mosaic(photo)

    ## Mosaic of the photo
    similarity = builder.compute_similarity(photo, mosaic)

    ## Mosaic has the default size and the metrics are present
    assert mosaic.shape == (512, 512, 3)

    ## Similarity metrics
    assert set(similarity) == {"MSE", "PSNR", "SSIM"}


## Test 20 -> Quadrant matching is closer to a real photo than mean matching
def test_quadrant_mode_is_no_worse_than_mean_on_a_photo():
    """On a real photo, quadrant features reach an MSE not larger than mean features."""
    ## Real sample photo, same tiles, two matching modes
    sample = PROJECT_ROOT / "sample_images" / "sample_image_joaquin_1.jpeg"

    ## All three metrics are present
    manager = TileManager(tile_directory = PROJECT_ROOT / "tiles")

    ## Real tile library
    errors = {}

    ## Errors of every mode
    for mode in ("mean", "quadrant"):
        ## Builder of this mode
        builder = MosaicBuilder(manager, grid_size = 16, image_size = 256, match_mode = mode)

        ## Store the MSE of the mode
        errors[mode] = builder.compute_similarity(sample, builder.create_mosaic(sample))["MSE"]

    ## Quadrant matching sees structure inside the cell
    assert errors["quadrant"] <= errors["mean"]


## Test 21 -> Configuration errors are caught at construction
def test_builder_configuration_errors():
    """Bad grids, sizes and options are reported before any image is processed."""
    ## Shared tile manager
    manager = TileManager(tile_set = "Solid")

    ## A grid that does not divide the image, zero, negative, boolean, huge and malformed grids
    with pytest.raises(InvalidGridError, match = "divide"):
        ## A grid that does not divide the image is rejected
        MosaicBuilder(manager, grid_size = 7, image_size = 512)

    ## Try invalid grids
    for bad in (0, -4, True, 10000, (4, 4, 4), "32"):
        ## Each invalid grid is rejected
        with pytest.raises(InvalidGridError):
            ## Each invalid grid is rejected
            MosaicBuilder(manager, grid_size = bad)

    ## Unknown options and the wrong type of tile manager
    with pytest.raises(MosaicError):
        ## An unknown matching mode is rejected
        MosaicBuilder(manager, match_mode = "median")

    ## An unknown matching mode is rejected
    with pytest.raises(MosaicError):
        ## An unknown strategy is rejected
        MosaicBuilder(manager, match_algorithm = "magic")

    ## An unknown strategy is rejected
    with pytest.raises(MosaicError):
        ## A path is not a tile manager
        MosaicBuilder("tiles/")

    ## The quadrant mode needs even cells (512 / 64 = 8 is fine, 504 / 56 = 9 is not)
    MosaicBuilder(manager, grid_size = 64, image_size = 512, match_mode = "quadrant")

    ## Odd cells are rejected in quadrant mode
    with pytest.raises(MosaicError, match = "even"):
        ## Odd cells are rejected in quadrant mode
        MosaicBuilder(manager, grid_size = 56, image_size = 504, match_mode = "quadrant")


## Test 22 -> Batch processing
def test_batch_matches_single_calls(photo):
    """Batch processing returns the same mosaics as calling create_mosaic one by one."""
    ## Three different inputs
    inputs = [photo, photo[::-1], photo[:, ::-1]]

    ## Builder for a small grid
    builder = MosaicBuilder(TileManager(tile_set = "Diagonal"), grid_size = 8, image_size = 64)

    ## Same results in the same order
    batch = builder.create_mosaics_batch(inputs)

    ## Batch results equal the single results
    assert all(np.array_equal(mosaic, builder.create_mosaic(image)) for mosaic, image in zip(batch, inputs))


## Test 23 -> Stage timings are reported
def test_build_reports_stage_timings(photo):
    """Every stage appears in the timings and their sum equals the total."""
    ## Build with segmentation
    builder = MosaicBuilder(TileManager(tile_set = "Diagonal"), grid_size = 8, image_size = 64)

    ## Timings of one build
    timings = builder.build(photo).timings

    ## Stage names and consistency
    assert {"prepare", "grid", "match", "assemble", "segment", "total"} <= set(timings)

    ## The total equals the sum of the stages
    assert timings["total"] == pytest.approx(sum(v for k, v in timings.items() if k != "total"))


## ********** Metrics **********
## Test 24 -> Metrics of identical and different images
def test_metrics_basic_values():
    """MSE is zero and SSIM one for identical images; MSE is exact for a constant offset."""
    ## Identical images
    image = np.random.default_rng(5).integers(0, 256, size = (32, 32, 3), dtype = np.uint8)

    ## Identical images have zero error
    assert metrics.mean_squared_error(image, image) == 0.0

    ## Identical images have infinite PSNR
    assert metrics.peak_signal_noise_ratio(image, image) == float("inf")

    ## Identical images have SSIM of one
    assert metrics.structural_similarity_index(image, image) == pytest.approx(1.0)

    ## A constant difference of 2 gives an MSE of 4
    shifted = np.clip(image.astype(int) + 2, 0, 255).astype(np.uint8)

    ## A constant offset of 10 gives an MSE of 100
    assert metrics.mean_squared_error(np.full((4, 4, 3), 10, dtype = np.uint8), np.full((4, 4, 3), 12, dtype = np.uint8)) == 4.0

    ## Different images have a positive error
    assert metrics.mean_squared_error(image, shifted) > 0


## Test 25 -> Metrics reject different shapes
def test_metrics_shape_mismatch():
    """Images with different shapes cannot be compared."""
    ## Different shapes
    with pytest.raises(MosaicError):
        ## Images of different shapes are rejected
        metrics.mean_squared_error(np.zeros((4, 4, 3)), np.zeros((5, 5, 3)))

    ## Vectors of different lengths are rejected
    with pytest.raises(MosaicError):
        ## Vectors of different lengths are rejected
        metrics.tile_agreement(np.zeros(3), np.zeros(4))


## Test 26 -> Tile agreement
def test_tile_agreement():
    """The fraction of equal assignments is computed correctly."""
    ## Three of four equal
    assert metrics.tile_agreement(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 0])) == 0.75


## ********** Utilities **********
## Test 27 -> Timing helpers
def test_timing_helpers():
    """time_callable summarizes runs and format_seconds picks a readable unit."""
    ## Five runs of a trivial function
    summary = time_callable(lambda: None, repeats = 5, warmup = 1)

    ## Summary of five runs
    assert len(summary["runs"]) == 5 and summary["mean"] >= 0

    ## Invalid repetition counts are rejected
    with pytest.raises(MosaicError):
        ## Zero repetitions are rejected
        time_callable(lambda: None, repeats = 0)

    ## Units
    assert format_seconds(2.5) == "2.50 s"

    ## Milliseconds are written with two decimals
    assert format_seconds(0.0123) == "12.30 ms"

    ## Microseconds are written with two decimals
    assert format_seconds(0.0000123) == "12.30 us"

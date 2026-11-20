# Image Mosaic Generator - Lab 8 (profiled, optimized and refactored)

By Joaquin Saldarriaga (NUID: 002597882) - CS5130, Northeastern University

Fast, modular photo-mosaic generator. It is the Lab 1 project after three steps: **profile it, make it fast, make it maintainable.**

* **Live demo:** _add the Streamlit Community Cloud link here after deploying_
* **Demo video:** _add the link here_

## Highlights

* **52x** faster than the Lab 1 code at 512x512 / 32x32 (target: 20x), with **pixel-identical** output (36 / 36 cases)
* Vectorized core (no loops over cells or pixels), cached tiles / features / k-d trees, a single resize per image
* Modular package, custom exceptions, 41 unit tests, a Streamlit app and a profiling notebook

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux / macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

For the notebook and the tests also run `pip install -r requirements-dev.txt`.

## Usage

### Python API

```python
import numpy as np
from PIL import Image
from mosaic_generator import MosaicBuilder, TileManager

## The tiles, their features and their k-d trees are cached inside the manager
manager = TileManager(tile_directory = "tiles")
builder = MosaicBuilder(manager, grid_size = 32, image_size = 512)

photo = np.asarray(Image.open("sample_images/sample_image_joaquin_1.jpeg").convert("RGB"))
mosaic = builder.create_mosaic(photo)                          # fast path: mosaic only
result = builder.build(photo)                                  # prepared, mosaic, indices, segmentation, timings
print(builder.compute_similarity(result.prepared, result.mosaic))   # MSE, PSNR, SSIM on demand
```

### Web app

```bash
streamlit run app.py
```

### Tests and notebook

```bash
pytest -q
jupyter notebook profiling_analysis.ipynb
python generate_tiles.py        # regenerates the 216-tile library in tiles/
```

## Project structure

```
Lab8_Mosaic_Optimization/
|-- app.py                       Streamlit application
|-- generate_tiles.py            builds the 216 color tiles
|-- profiling_analysis.ipynb     cProfile + line_profiler + optimization ladder
|-- mosaic_generator/
|   |-- config.py                constants and limits
|   |-- utils.py                 exceptions, validation, timers
|   |-- image_processor.py       prepare, cell means, segmentation, assembly
|   |-- tile_manager.py          tile loading, LRU cache, k-d tree / brute search
|   |-- mosaic_builder.py        MosaicBuilder, MosaicResult
|   `-- metrics.py               MSE, PSNR, SSIM, tile agreement
|-- tests/test_mosaic.py         41 tests (correctness vs. Lab 1, errors, cache)
|-- baseline_lab1/               Lab 1 code, kept as the reference
|-- tiles/  sample_images/  results/
```

## What was optimized

* **Request path:** the Lab 1 `create_mosaic` also ran SSIM (53 % of the call) and a loops-vs-vectorized benchmark (40 %) in every request; both are now on demand / separate tools
* **Cell means:** two contiguous integer reductions (uint16, uint32) instead of a strided float32 mean - same output, about 13x faster
* **Tiles:** loaded once, cached per cell size (LRU), features and k-d tree cached with them
* **Matching:** one vectorized nearest-neighbour query (`cKDTree`, or a matrix-product search for very small libraries - chosen by `auto`)
* **Resize:** crop then resize once (identical to Lab 1)
* **Segmentation:** built only when requested

## Benchmarks

Measured with `3.13`, NumPy 2.5.3, SciPy 1.18.1, 2 CPUs, Linux. Re-run the notebook on your machine to refresh them.

| Image size | Grid size | Baseline time (s) | Lab 8 time (s) | Speedup |
|---|---|---|---|---|
| 256×256 | 16×16 | 0.0498 | 0.00063 | 79.6x |
| 512×512 | 32×32 | 0.1196 | 0.00229 | 52.2x |
| 1024×1024 | 64×64 | 0.4951 | 0.00835 | 59.3x |

Speedup of the full Lab 8 request over the Lab 1 `create_mosaic` call, all nine combinations:

| Image | Grid | Lab 1 (s) | Lab 8 (s) | Speedup | Speedup if Lab 8 also computes the metrics | Speedup of the core only |
|---|---|---|---|---|---|---|
| 256x256 | 16x16 | 0.0498 | 0.00063 | 79.6x | 3.6x | 3.0x |
| 256x256 | 32x32 | 0.0463 | 0.00078 | 59.5x | 3.4x | 2.5x |
| 256x256 | 64x64 | 0.0542 | 0.00157 | 34.6x | 3.6x | 1.7x |
| 512x512 | 16x16 | 0.1196 | 0.00222 | 53.9x | 2.1x | 3.2x |
| 512x512 | 32x32 | 0.1196 | 0.00229 | 52.2x | 2.0x | 3.5x |
| 512x512 | 64x64 | 0.1193 | 0.00273 | 43.7x | 1.9x | 3.1x |
| 1024x1024 | 16x16 | 0.5016 | 0.00806 | 62.2x | 1.4x | 3.7x |
| 1024x1024 | 32x32 | 0.4791 | 0.00919 | 52.1x | 1.4x | 3.1x |
| 1024x1024 | 64x64 | 0.4951 | 0.00835 | 59.3x | 1.4x | 3.8x |

How to read the three speedup columns:

* **Speedup** compares the whole Lab 1 call (which includes SSIM and the benchmark) with the Lab 8 mosaic request
* **With metrics** compares the same Lab 1 call with a Lab 8 request that also computes MSE / PSNR / SSIM - SSIM (scikit-image) is unchanged and dominates
* **Core only** compares the Lab 1 stages without SSIM and without the benchmark with Lab 8 - the gain of the algorithmic changes alone

## Correctness

* 36 / 36 mosaics and segmentations are pixel-identical to Lab 1 (MSE = 0.0, 100 % of tiles match)
* The k-d tree and the brute-force search only differ on exact ties

## Known limits

* The first request builds the cache (a few ms); later ones reuse it
* The `quadrant` feature needs an even cell size
* Large photos spend most of the time in the Lanczos resize

## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Tile Library Generator File

"""Generate the library of image tiles stored in the 'tiles' folder.

Process:
    * Sample the RGB cube on a regular grid (6 levels per channel = 216 colors)
    * Give every color a texture (gradient, diagonal, checks, stripes or rings)
    * Save every tile as a small PNG file
"""

## ********** Import Required Libraries **********
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


## ********** Tile Parameters **********
## Side of every generated tile in pixels
TILE_SIDE = 32

## Levels per color channel (6 x 6 x 6 = 216 colors)
LEVELS = np.linspace(20, 235, 6).astype(int)

## Strength of the texture, as a fraction of the full intensity range
TEXTURE_STRENGTH = 0.16


## Function 1 -> Build the texture patterns
def texture_patterns(side: int) -> dict[str, np.ndarray]:
    """Return zero-mean texture patterns with values in [-1, 1].

    Args:
        * side (int) -> Side of the square pattern in pixels

    Returns:
        * dict -> Pattern name mapped to a float32 array with shape (side, side)
    """
    ## Normalized coordinates in [0, 1]
    coordinates = np.linspace(0, 1, side, dtype = np.float32)

    ## Coordinate grids
    x, y = np.meshgrid(coordinates, coordinates)

    ## Raw patterns with different structures
    raw = {
        "hgrad": x,
        "vgrad": y,
        "diag": (x + y) / 2,
        "check": ((np.floor(x * 4) + np.floor(y * 4)) % 2).astype(np.float32),
        "stripe": (np.floor(x * 8) % 2).astype(np.float32),
        "ring": np.sin(np.hypot(x - 0.5, y - 0.5) * 18).astype(np.float32),
    }

    ## Center every pattern on zero and scale it to [-1, 1] so that the tile keeps its base color on average
    patterns = {}

    ## Process every pattern
    for name, pattern in raw.items():
        ## Center the pattern on zero
        centered = pattern - pattern.mean()

        ## Scale to [-1, 1]
        patterns[name] = centered / np.abs(centered).max()

    ## Return the patterns
    return patterns


## Function 2 -> Write the tile files
def generate_tiles(output_directory: Path) -> int:
    """Create the tile library on disk.

    Args:
        * output_directory (Path) -> Folder that receives the PNG files

    Returns:
        * int -> Number of tiles written
    """
    ## Create the folder if needed and build the patterns once
    output_directory.mkdir(parents = True, exist_ok = True)

    ## Build the patterns once
    patterns = texture_patterns(TILE_SIDE)

    ## Pattern names in a fixed order
    pattern_names = list(patterns)

    ## One tile for every (red, green, blue) level combination
    count = 0

    ## Loop over the red levels
    for red in LEVELS:
        ## Loop over the green levels
        for green in LEVELS:
            ## Loop over the blue levels
            for blue in LEVELS:

                ## Rotate through the patterns so that neighbouring colors get different textures
                name = pattern_names[count % len(pattern_names)]

                ## Base color of this tile
                base = np.array([red, green, blue], dtype = np.float32)

                ## Add the texture around the base color and save as an 8-bit PNG
                tile = base + patterns[name][..., None] * 255.0 * TEXTURE_STRENGTH

                ## Convert the tile to an 8-bit image
                image = Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8))

                ## Write the PNG file
                image.save(output_directory / f"tile_{count:03d}_{name}_r{red:03d}_g{green:03d}_b{blue:03d}.png")

                ## Count the tile
                count += 1

    ## Return the number of tiles
    return count


## ********** Command Line Entry Point **********
if __name__ == "__main__":

    ## Optional output folder argument
    parser = argparse.ArgumentParser(description = "Generate the tile library")

    ## Output folder option
    parser.add_argument("--output", type = Path, default = Path(__file__).parent / "tiles")

    ## Read the command line
    arguments = parser.parse_args()

    ## Write the files and report
    written = generate_tiles(arguments.output)

    ## Report the result
    print(f"Wrote {written} tiles to {arguments.output}")

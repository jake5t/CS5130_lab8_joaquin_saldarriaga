## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Package Initialization File

"""Mosaic generator package: fast, modular photo mosaics from small tiles.

Public API:
    * TileManager -> loads and caches tiles, features and nearest-tile search
    * MosaicBuilder -> builds mosaics and computes similarity metrics
"""

## ********** Import the Public Classes **********
from .mosaic_builder import MosaicBuilder, MosaicResult
from .tile_manager import TileManager
from .utils import InvalidGridError, InvalidImageError, MosaicError, TileLoadError

## ********** Names Exported by 'from mosaic_generator import *' **********
__all__ = [
    "MosaicBuilder",
    "MosaicResult",
    "TileManager",
    "MosaicError",
    "InvalidImageError",
    "InvalidGridError",
    "TileLoadError",
]

## ********** Package Version **********
__version__ = "1.0.0"

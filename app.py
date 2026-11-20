## By - Joaquin Saldarriaga (NUID: 002597882)
## Challenge on Codes Northeastern University
## Lab  8 -> Week 10 - Streamlit Application

"""Interactive web application of the optimized Image Mosaic Generator.

Process:
    * Load the tile library once and keep it in memory (cached tiles, features and k-d trees)
    * Let the user upload a photo and choose the grid, the tiles and the matching mode
    * Build the mosaic in a few milliseconds and show the time of every stage
    * Compute the similarity metrics (MSE, PSNR, SSIM) only when they are asked for
"""

## ********** Import Required Libraries **********
import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from mosaic_generator import MosaicBuilder, MosaicError, TileManager
from mosaic_generator import config
from mosaic_generator.utils import format_seconds


## ********** Constants of the Application **********
## Folder of the project, used to find the tiles and the sample images
PROJECT = Path(__file__).resolve().parent

## Folder with the 216 pre-generated color tiles
TILE_FOLDER = PROJECT / "tiles"

## Folder with the sample photos
SAMPLE_FOLDER = PROJECT / "sample_images"


## This function loads one tile source once and keeps it in memory for every request
@st.cache_resource(show_spinner = "Loading the tiles...")
def load_tile_manager(source: str) -> TileManager:
    """ Create the tile manager of a source, shared by all the requests

    Args:
     * source (str) -> 'Folder: tiles/' or the name of a procedural family

    Returns:
     * TileManager -> manager with its cache of tiles, features and k-d trees
    """

    ## The pre-generated library comes from the folder
    if source.startswith("Folder"):
        ## Return the manager of the folder
        return TileManager(tile_directory = TILE_FOLDER)

    ## Any other source is a procedural family
    return TileManager(tile_set = source)


## This function builds the page: controls on the left, images in the main area
def main() -> None:
    """ Draw the whole page and run the pipeline when a photo is available """

    ## Page title and layout
    st.set_page_config(page_title = "Image Mosaic Generator", layout = "wide")

    ## Heading and short description
    st.title("Image Mosaic Generator (optimized)")

    ## Short description
    st.caption("Lab 8 - vectorized, cached and modular version of the Lab 1 mosaic generator")

    ## ********** Sidebar: Controls **********
    ## Options of the tile source: the folder first, then the procedural families
    sources = [f"Folder: {TILE_FOLDER.name}/"] + list(config.PROCEDURAL_TILE_SETS)

    ## Selection of the tile source
    source = st.sidebar.selectbox("Tiles", sources)

    ## Selection of the number of cells per side
    grid = st.sidebar.select_slider("Grid size (cells per side)", options = list(config.SUPPORTED_GRID_SIZES), value = config.DEFAULT_GRID_SIZE)

    ## Selection of the side of the working image
    size = st.sidebar.select_slider("Image size (pixels)", options = [256, 512, 1024], value = config.DEFAULT_IMAGE_SIZE)

    ## Selection of the matching feature
    mode = st.sidebar.radio("Matching feature", list(config.MATCH_MODES), help = "'mean' compares the mean color, 'quadrant' compares the four quadrants of every cell")

    ## Selection of the search strategy
    algorithm = st.sidebar.radio("Search", list(config.MATCH_ALGORITHMS), help = "'auto' chooses the k-d tree or the brute force search by the number of tiles")

    ## ********** Input Image **********
    ## Sample photos available in the project
    samples = sorted(SAMPLE_FOLDER.glob("*.jp*g")) if SAMPLE_FOLDER.is_dir() else []

    ## Upload widget
    uploaded = st.sidebar.file_uploader("Upload a photo", type = ["png", "jpg", "jpeg", "webp"])

    ## Sample selector, used when nothing is uploaded
    chosen = st.sidebar.selectbox("...or use a sample", samples, format_func = lambda path: path.name) if samples else None

    ## An upload has priority over the sample
    photo_source = uploaded if uploaded is not None else chosen

    ## Nothing to process yet
    if photo_source is None:
        ## Tell the user what to do
        st.info("Upload a photo to start.")

        ## Stop the page here
        return

    ## Read the photo as an RGB image
    photo = Image.open(photo_source).convert("RGB")

    ## ********** Pipeline **********
    ## Tile manager of the source (cached between requests)
    manager = load_tile_manager(source)

    ## Try to build the mosaic and report any invalid combination
    try:
        ## Builder with the options of the sidebar
        builder = MosaicBuilder(manager, grid_size = grid, image_size = size, match_mode = mode, match_algorithm = algorithm)

        ## Build everything that the page shows
        result = builder.build(np.asarray(photo))
    except MosaicError as error:
        ## Show the message of the error instead of a traceback
        st.error(str(error))

        ## Stop the page here
        return

    ## ********** Output **********
    ## Three columns: prepared image, segmentation and mosaic
    columns = st.columns(3)

    ## Images and their titles
    for column, title, image in zip(columns, ("Preprocessed", "Segmented", "Mosaic"), (result.prepared, result.segmented, result.mosaic)):
        ## Title of the column
        column.subheader(title)

        ## Image of the column
        column.image(image, width = "stretch")

    ## Time of every stage of this request
    st.subheader("Time of this request")

    ## Metrics in columns, one per stage
    timing_columns = st.columns(len(result.timings))

    ## Show every stage with a readable unit
    for column, (stage, seconds) in zip(timing_columns, result.timings.items()):
        ## Name and duration of the stage
        column.metric(stage, format_seconds(seconds))

    ## Tile cache counters, to show that the library is not reloaded
    st.caption(f"Tile cache: {manager.cache_hits} hits / {manager.cache_misses} misses - {manager.n_tiles} tiles loaded")

    ## ********** Similarity Metrics (on demand) **********
    ## The SSIM is the slowest metric, so it only runs on request
    if st.button("Compute similarity metrics"):
        ## Metrics between the preprocessed image and the mosaic
        similarity = builder.compute_similarity(result.prepared, result.mosaic)

        ## Three columns, one per metric
        metric_columns = st.columns(3)

        ## Show every metric
        for column, (name, value) in zip(metric_columns, similarity.items()):
            ## Name and value of the metric
            column.metric(name, f"{value:.4f}")

    ## ********** Download **********
    ## Encode the mosaic as a PNG in memory
    buffer = io.BytesIO()

    ## Save the PNG in the buffer
    Image.fromarray(result.mosaic).save(buffer, format = "PNG")

    ## Download button
    st.download_button("Download the mosaic (PNG)", buffer.getvalue(), file_name = "mosaic.png", mime = "image/png")


## ********** Run the Application **********
## Streamlit executes this file as a script
main()

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import rasterio
from rasterio.windows import Window

from satchip import models


def make_grid_from_reference(reference: Path, chip_size = 256) -> list[models.GridCell]:
    grid = []

    with rasterio.open(reference) as ref:
        n_cols = ref.width // chip_size
        n_rows = ref.height // chip_size

        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(col * chip_size, row * chip_size, chip_size, chip_size)
                bounds = ref.window_bounds(window)

                cell_id = f"{row:03d}.{col:03d}"

                grid.append(models.GridCell(cell_id, bounds))

    return grid


def chip_data(grid: list[models.GridCell], layers: Iterable[Path], output_path: Path):
    chips = {}

    for layer in layers:
        with rasterio.open(layer) as src:
            for tile_id, bounds in grid:
                window = src.window(*bounds)
                window = Window(
                    round(window.col_off),
                    round(window.row_off),
                    round(window.width),
                    round(window.height),
                )

                data = src.read(window=window)

                chip_meta = src.meta.copy()
                chip_meta.update(
                    {
                        "width": window.width,
                        "height": window.height,
                        "transform": src.window_transform(window),
                    }
                )

                chip_name = f"{tile_id}.{layer.name}"
                chip_path = output_path / chip_name

                with rasterio.open(chip_path, "w", **chip_meta) as dst:
                    dst.write(data)

                chips[tile_id] = chip_path

    return chips


def filter_chips(chips, modality):
    if modality.id == 'HLS':
        filtered_chips = filter_hls_chips(chips)
    elif modality.id == 'RTC':
        filtered_chips = filter_rtc_chips(chips)

    return filtered_chips


def filter_hls_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip["Fmask"]) as ds:
            qc = clear_px_Fmask(ds.read(1))

        with rasterio.open(chip["EVENT"]) as ds:
            event = ds.read(1)

        ny, nx = qc.shape
        n_px = 1.0 * ny * nx

        # cloud-free pixels (0 clear, 1 cloud, 255 nodata)
        n_cf = len(np.where(qc == 0)[0])
        pct_cf = 100.0 * (n_cf / n_px)

        # event pixels
        n_ev = len(np.where(event > 0)[0])

        # pct of chip in event
        pct_ev = 100.0 * (n_ev / n_px) if n_ev > 0 else 0

        if pct_cf > 95 and pct_ev > 1:
            good_chips.append(chip)

    return good_chips


def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # clip the data to be in the range of cmin to cmax
    arr = np.clip(arr, cmin, cmax)
    high = float(high)
    low = float(low)
    cmax = float(cmax)
    cmin = float(cmin)
    m = (high - low) / (cmax - cmin)  # slope
    b = high - (m * cmax)  # intercept
    arr = np.uint8((m * arr) + b)
    return arr


def clear_px_Fmask(Fmask: np.ndarray) -> np.ndarray:
    fmask_clear = np.array([
        0, 4, 16, 20, 32, 36, 48, 52,
        64, 68, 80, 84, 96, 100, 112, 116,
        128, 132, 144, 148, 160, 164, 176, 180,
        192, 196, 208, 212, 224, 228, 240, 244
    ], dtype=Fmask.dtype)

    cloudmask = np.ones_like(Fmask, dtype=np.uint8)
    cloudmask[np.isin(Fmask, fmask_clear)] = 0
    cloudmask[Fmask == 255] = 255

    return cloudmask


def filter_rtc_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip["BANDS"]) as ds:
            rtc_data = ds.read()

        with rasterio.open(chip["EVENT"]) as ds:
            event_mask = ds.read(1)

        has_nan_pixels = np.isnan(rtc_data).sum() > 0

        num_pixels = event_mask.size
        num_event_pixels = np.count_nonzero(event_mask > 0)

        pct_pixels_over_event = 100.0 * (num_event_pixels / num_pixels)
        data_overlaps_event = pct_pixels_over_event > 1

        if not has_nan_pixels and data_overlaps_event:
            good_chips.append(chip)

    return good_chips

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from satchip import models


def make_grid_from_reference(reference: Path, chip_size: int = 256) -> list[models.GridCell]:
    grid = []

    with rasterio.open(reference) as ref:
        n_cols = ref.width // chip_size
        n_rows = ref.height // chip_size

        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(col * chip_size, row * chip_size, chip_size, chip_size)
                bounds = ref.window_bounds(window)

                cell_id = f'{row:03d}.{col:03d}'

                grid.append(models.GridCell(cell_id, bounds))

    return grid


def chip_data(grid: list[models.GridCell], layer: Path, output_path: Path) -> list[models.Chips]:
    output_path.mkdir(exist_ok=True, parents=True)

    chips = []

    with rasterio.open(layer) as src:
        for grid_cell in grid:
            window = src.window(*grid_cell.bounds)
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
                    'width': window.width,
                    'height': window.height,
                    'transform': src.window_transform(window),
                }
            )

            chip_name = f'{grid_cell.id}.{layer.name}'
            chip_path = output_path / chip_name

            with rasterio.open(chip_path, 'w', **chip_meta) as dst:
                dst.write(data)

            chip = models.Chip(grid_cell.id, chip_path)
            chips.append(chip)

    return chips


def make_chip_stacks(
    data_chips: list[models.Chip],
    validation_mask_chips: list[models.Chip],
    label_chips: list[models.Chip],
    modality: models.Modality,
) -> list[models.ChipStack]:
    chip_stacks = []

    for data, mask, label in zip(
        sorted(data_chips, key=lambda c: c.id),
        sorted(validation_mask_chips, key=lambda c: c.id),
        sorted(label_chips, key=lambda c: c.id),
    ):
        chip_stack = models.ChipStack(
            id=data.id,
            data=data.path,
            validation_mask=mask.path,
            label=label.path,
            modality=modality,
        )

        chip_stacks.append(chip_stack)

    return chip_stacks


def filter_chips(chip_stacks: list[models.ChipStack]) -> list[models.ChipStack]:
    good_chips = []

    for chip_stack in chip_stacks:
        if 'HLS' in chip_stack.modality['id']:
            is_good_chip = is_good_hls_chip(chip_stack)
        elif 'RTC' in chip_stack.modality['id']:
            is_good_chip = is_good_rtc_chip(chip_stack)

        if is_good_chip:
            good_chips.append(chip_stack)

    return good_chips


def is_good_hls_chip(chip_stack: models.ChipStack) -> bool:
    with rasterio.open(chip_stack.validation_mask) as ds:
        qc = clear_px_Fmask(ds.read(1))

    with rasterio.open(chip_stack.label) as ds:
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

    return pct_cf > 95 and pct_ev > 1


def clear_px_Fmask(Fmask: np.ndarray) -> np.ndarray:
    fmask_clear = np.array(
        [
            0,
            4,
            16,
            20,
            32,
            36,
            48,
            52,
            64,
            68,
            80,
            84,
            96,
            100,
            112,
            116,
            128,
            132,
            144,
            148,
            160,
            164,
            176,
            180,
            192,
            196,
            208,
            212,
            224,
            228,
            240,
            244,
        ],
        dtype=Fmask.dtype,
    )

    cloudmask = np.ones_like(Fmask, dtype=np.uint8)
    cloudmask[np.isin(Fmask, fmask_clear)] = 0
    cloudmask[Fmask == 255] = 255

    return cloudmask


def is_good_rtc_chip(chip_stack: models.ChipStack) -> bool:
    with rasterio.open(chip_stack.data) as ds:
        rtc_data = ds.read()

    with rasterio.open(chip_stack.label) as ds:
        event_mask = ds.read(1)

    has_nan_pixels = np.isnan(rtc_data).sum() > 0

    num_pixels = event_mask.size
    num_event_pixels = np.count_nonzero(event_mask > 0)

    pct_pixels_over_event = 100.0 * (num_event_pixels / num_pixels)
    data_overlaps_event = pct_pixels_over_event > 1

    return not has_nan_pixels and data_overlaps_event

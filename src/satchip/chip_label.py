import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr
from tqdm import tqdm

from satchip import utils
from satchip.chip_xr_base import create_dataset_chip
from satchip.terra_mind_grid import TerraMindGrid


def is_valuable(chip: np.ndarray) -> bool:
    vals = list(np.unique(chip))
    return not vals == [0]


def chip_labels(label_path: Path, date: datetime, chip_dir: Path) -> list[Path]:
    label_dir = chip_dir / 'LABEL'
    label_dir.mkdir(parents=True, exist_ok=True)
    label = xr.open_dataarray(label_path)
    bbox = utils.get_epsg4326_bbox(label.rio.bounds(), label.rio.crs.to_epsg())
    tm_grid = TerraMindGrid(latitude_range=(bbox[1], bbox[3]), longitude_range=(bbox[0], bbox[2]))
    chip_paths = []
    for tm_chip in tqdm(tm_grid.terra_mind_chips):
        chip = label.rio.reproject(
            dst_crs=f'EPSG:{tm_chip.epsg}',
            resampling=rio.enums.Resampling(1),  # type: ignore
            transform=tm_chip.rio_transform,
            shape=(tm_chip.nrow, tm_chip.ncol),
        )
        chip_array = chip.data[0]
        chip_array[np.isnan(chip_array)] = 0
        chip_array = np.round(chip_array).astype(np.int16)
        if is_valuable(chip_array):
            dataset = create_dataset_chip(chip_array.reshape(1, *chip_array.shape), tm_chip, date, ['LABEL'])
            chip_path = label_dir / f'{label_path.stem}_{tm_chip.name}.zarr.zip'
            utils.save_chip(dataset, chip_path)
            chip_paths.append(chip_path)
    return chip_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=str, help='Path to the label image')
    parser.add_argument('date', type=str, help='Date and time of the image in ISO format (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--chipdir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()
    args.labelpath = Path(args.labelpath)
    args.date = datetime.fromisoformat(args.date)
    args.chipdir = Path(args.chipdir)
    chip_labels(args.labelpath, args.date, args.chipdir)


if __name__ == '__main__':
    main()

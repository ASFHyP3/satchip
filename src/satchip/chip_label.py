import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio as rio
import xarray as xr
from tqdm import tqdm

import satchip
from satchip import utils
from satchip.terra_mind_grid import TerraMindChip, TerraMindGrid


def get_overall_bounds(bounds: list) -> list:
    minx = min([b[0] for b in bounds])
    miny = min([b[1] for b in bounds])
    maxx = max([b[2] for b in bounds])
    maxy = max([b[3] for b in bounds])
    return [minx, miny, maxx, maxy]


def is_valuable(chip: np.ndarray) -> bool:
    vals = list(np.unique(chip))
    return not vals == [0]


def create_dataset_chip(chip_array: np.ndarray, tm_chip: TerraMindChip, bands: list[str], date: datetime) -> xr.Dataset:
    x = tm_chip.minx + (np.arange(tm_chip.nrow) + 0.5) * tm_chip.xres
    y = tm_chip.maxy + (np.arange(tm_chip.ncol) + 0.5) * tm_chip.yres
    coords = {'time': np.array([date]), 'band': np.array(bands), 'y': y, 'x': x}
    dataset = xr.Dataset(attrs={'date_created': date.isoformat(), 'satchip_version': satchip.__version__})
    dataset.attrs['bounds'] = tm_chip.bounds
    dataset = dataset.assign_coords(sample=tm_chip.name)
    dataset = dataset.rio.write_crs(f'EPSG:{tm_chip.epsg}')
    shape = (1, 1, tm_chip.ncol, tm_chip.nrow)
    dataset['bands'] = xr.DataArray(chip_array.reshape(*shape), coords=coords, dims=['time', 'band', 'y', 'x'])
    dataset['center_lat'] = xr.DataArray(tm_chip.center[1])
    dataset['center_lon'] = xr.DataArray(tm_chip.center[0])
    dataset['crs'] = xr.DataArray(tm_chip.epsg)
    utils.check_spec(dataset)
    return dataset


def chip_labels(label_path: Path, date: datetime, output_dir: Path) -> list[Path]:
    label_dir = output_dir / 'LABEL'
    label_dir.mkdir(parents=True, exist_ok=True)
    label = xr.open_dataarray(label_path)
    bbox = utils.get_epsg4326_bbox(label.rio.bounds(), label.rio.crs.to_epsg())
    tm_grid = TerraMindGrid(latitude_range=(bbox[1], bbox[3]), longitude_range=(bbox[0], bbox[2]))
    output_paths = []
    for tm_chip in tqdm(tm_grid.terra_mind_chips):
        chip = label.rio.reproject(
            dst_crs=f'EPSG:{tm_chip.epsg}',
            resampling=rio.enums.Resampling(1),
            transform=tm_chip.rio_transform,
            shape=(tm_chip.nrow, tm_chip.ncol),
        )
        chip_array = chip.data[0]
        chip_array[np.isnan(chip_array)] = 0
        chip_array = np.round(chip_array).astype(np.int16)
        if is_valuable(chip_array):
            dataset = create_dataset_chip(chip_array, tm_chip, ['label'], date)
            output_path = label_dir / f'{label_path.stem}_{tm_chip.name}.zarr.zip'
            utils.save_chip(dataset, output_path)
            output_paths.append(output_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=str, help='Path to the label image')
    parser.add_argument('date', type=str, help='Date and time of the image in ISO format (YYYY-MM-DDTHH:MM:SS)')
    parser.add_argument('--outdir', default='.', type=str, help='Output directory for the chips')
    args = parser.parse_args()
    args.labelpath = Path(args.labelpath)
    args.date = datetime.fromisoformat(args.date)
    args.outdir = Path(args.outdir)
    chip_labels(args.labelpath, args.date, args.outdir)


if __name__ == '__main__':
    main()

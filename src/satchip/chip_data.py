import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from shapely.geometry import box
from tqdm import tqdm

from satchip import utils
from satchip.chip_hls import get_hls_data
from satchip.chip_hyp3s1rtc import get_rtc_paths_for_chips, get_s1rtc_chip_data
from satchip.chip_operas1rtc import get_operartc_data
from satchip.chip_sentinel2 import get_s2l2a_data
from satchip.terra_mind_grid import TerraMindChip, TerraMindGrid


def fill_missing_times(data_chip: xr.DataArray, times: np.ndarray) -> xr.DataArray:
    missing_times = np.setdiff1d(times, data_chip.time.data)
    missing_shape = (len(missing_times), len(data_chip.band), data_chip.y.size, data_chip.x.size)
    missing_data = xr.DataArray(
        np.full(missing_shape, 0, dtype=data_chip.dtype),
        dims=('time', 'band', 'y', 'x'),
        coords={
            'time': missing_times,
            'band': data_chip.band.data,
            'y': data_chip.y.data,
            'x': data_chip.x.data,
        },
    )
    return xr.concat([data_chip, missing_data], dim='time').sortby('time')


def get_chip(label_path: Path) -> TerraMindChip:
    label_dataset = utils.load_chip(label_path)
    buffered = box(*label_dataset.bounds).buffer(0.1).bounds
    grid = TerraMindGrid([buffered[1], buffered[3]], [buffered[0], buffered[2]])  # type: ignore
    label_chip_name = label_dataset.sample.item()
    chip = [c for c in grid.terra_mind_chips if c.name == label_chip_name]
    assert len(chip) == 1, f'No TerraMind chip found for label {label_chip_name}'
    return chip[0]


def chip_data(
    chip: TerraMindChip,
    platform: str,
    opts: utils.ChipDataOpts,
    image_dir: Path,
) -> xr.Dataset:
    if platform == 'HYP3S1RTC':
        rtc_paths = opts['local_hyp3_paths'][chip.name]
        chip_dataset = get_s1rtc_chip_data(chip, rtc_paths)
    elif platform == 'S1RTC':
        chip_dataset = get_operartc_data(chip, image_dir, opts=opts)
    elif platform == 'S2L2A':
        chip_dataset = get_s2l2a_data(chip, image_dir, opts=opts)
    elif platform == 'HLS':
        chip_dataset = get_hls_data(chip, image_dir, opts=opts)
    else:
        raise Exception(f'Unknown platform {platform}')

    return chip_dataset


def create_chips(
    label_paths: list[Path],
    platform: str,
    date_start: datetime,
    date_end: datetime,
    strategy: str,
    max_cloud_pct: int,
    chip_dir: Path,
    image_dir: Path,
) -> list[Path]:
    platform_dir = chip_dir / platform
    platform_dir.mkdir(parents=True, exist_ok=True)

    opts: utils.ChipDataOpts = {'strategy': strategy, 'date_start': date_start, 'date_end': date_end}
    if platform in ['S2L2A', 'HLS']:
        opts['max_cloud_pct'] = max_cloud_pct

    chips = [get_chip(p) for p in label_paths]
    chip_names = [c.name for c in chips]
    if len(chip_names) != len(set(chip_names)):
        duplicates = [name for name, count in Counter(chip_names).items() if count > 1]
        msg = f'Duplicate sample locations not supported. Duplicate chips: {", ".join(duplicates)}'
        raise NotImplementedError(msg)
    chip_paths = [
        platform_dir / (x.with_suffix('').with_suffix('').name + f'_{platform}.zarr.zip') for x in label_paths
    ]
    if platform == 'HYP3S1RTC':
        rtc_paths_for_chips = get_rtc_paths_for_chips(chips, image_dir, opts)
        opts['local_hyp3_paths'] = rtc_paths_for_chips

    for chip, chip_path in tqdm(zip(chips, chip_paths), desc='Chipping labels'):
        dataset = chip_data(chip, platform, opts, image_dir)
        utils.save_chip(dataset, chip_path)
    return chip_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=Path, help='Path to the label directory')
    parser.add_argument(
        'platform', choices=['S1RTC', 'S2L2A', 'HLS', 'HYP3S1RTC'], type=str, help='Dataset to create chips for'
    )
    parser.add_argument('daterange', type=str, help='Inclusive date range to search for data in the format Ymd-Ymd')
    parser.add_argument('--maxcloudpct', default=100, type=int, help='Maximum percent cloud cover for a data chip')
    parser.add_argument('--chipdir', default='.', type=Path, help='Output directory for the chips')
    parser.add_argument(
        '--imagedir', default=None, type=Path, help='Output directory for image files. Defaults to chipdir/IMAGES'
    )
    parser.add_argument(
        '--strategy',
        default='BEST',
        choices=['BEST', 'ALL'],
        type=str,
        help='Strategy to use when multiple scenes are found (default: BEST)',
    )
    args = parser.parse_args()
    args.platform = args.platform.upper()
    assert 0 <= args.maxcloudpct <= 100, 'maxcloudpct must be between 0 and 100'
    date_start, date_end = [datetime.strptime(d, '%Y%m%d') for d in args.daterange.split('-')]
    assert date_start < date_end, 'start date must be before end date'
    label_paths = list(args.labelpath.glob('*.zarr.zip'))
    assert len(label_paths) > 0, f'No label files found in {args.labelpath}'

    if args.imagedir is None:
        args.imagedir = args.chipdir / 'IMAGES'

    create_chips(
        label_paths, args.platform, date_start, date_end, args.strategy, args.maxcloudpct, args.chipdir, args.imagedir
    )


if __name__ == '__main__':
    main()

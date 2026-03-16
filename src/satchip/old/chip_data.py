import argparse
import glob
from collections import Counter
from datetime import datetime
from pathlib import Path

import earthaccess
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


def get_chips(label_paths: list[Path]) -> list[TerraMindChip]:
    label_datasets = [utils.load_chip(label_path) for label_path in label_paths]
    bounds = utils.get_overall_bounds([ds.bounds for ds in label_datasets])

    buffered = box(*bounds).buffer(0.5).bounds
    grid = TerraMindGrid(latitude_range=(buffered[1], buffered[3]), longitude_range=(buffered[0], buffered[2]))
    grid_chips = {chip.name: chip for chip in grid.terra_mind_chips}

    chips = []
    for label_dataset in label_datasets:
        label_chip_name = label_dataset.sample.item()
        assert label_chip_name in grid_chips, f'No TerraMind chip found for label {label_chip_name}'
        chip = grid_chips[label_chip_name]
        chips.append(chip)

    return chips


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
    dates: utils.DateRange | list[datetime],
    strategy: str,
    max_cloud_pct: int,
    chip_dir: Path,
    image_dir: Path,
) -> list[Path]:
    platform_dir = chip_dir / platform
    platform_dir.mkdir(parents=True, exist_ok=True)

    opts: utils.ChipDataOpts = {'strategy': strategy, 'dates': dates}
    if platform in ['S2L2A', 'HLS']:
        opts['max_cloud_pct'] = max_cloud_pct

    if platform in ['S1RTC', 'HLS']:
        earthaccess.login()

    chips = get_chips(label_paths)
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

    for chip, chip_path in tqdm(list(zip(chips, chip_paths)), desc='Chipping labels'):
        dataset = chip_data(chip, platform, opts, image_dir)
        utils.save_chip(dataset, chip_path)
    return chip_paths


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')

    parser.add_argument('labels', type=str, help='Path or Glob pattern for label chips')

    parser.add_argument(
        'platform', choices=['S1RTC', 'S2L2A', 'HLS', 'HYP3S1RTC'], type=str, help='Dataset to create chips for'
    )

    date_group = parser.add_mutually_exclusive_group(required=True)

    date_group.add_argument(
        '--daterange',
        nargs=2,
        type=str,
        metavar=('START', 'END'),
        help='Inclusive date range in the format YYYY-mm-dd YYYY-mm-dd',
    )
    date_group.add_argument(
        '--dates', nargs='+', type=str, help='Space-separated list of specific dates in format YYYY-mm-dd'
    )

    parser.add_argument('--maxcloudpct', default=100, type=int, help='Maximum percent cloud cover for a data chip')
    parser.add_argument('--chipdir', default='.', type=Path, help='Output directory for the chips')
    parser.add_argument(
        '--imagedir', default=None, type=Path, help='Output directory for image files. Defaults to chipdir/IMAGES'
    )
    parser.add_argument(
        '--strategy',
        default='BEST',
        choices=['BEST', 'ALL', 'SPECIFIC'],
        type=str,
        help='Strategy to use when multiple scenes are found (default: BEST)',
    )
    args = parser.parse_args()

    args.platform = args.platform.upper()
    assert 0 <= args.maxcloudpct <= 100, 'maxcloudpct must be between 0 and 100'

    dates: utils.DateRange | list[datetime]
    if args.daterange:
        start_date = datetime.strptime(args.daterange[0], '%Y-%m-%d')
        end_date = datetime.strptime(args.daterange[1], '%Y-%m-%d')
        assert start_date < end_date, 'start date must be before end date'

        dates = utils.DateRange(start_date, end_date)
    else:
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in args.dates]
        args.strategy = 'SPECIFIC'

    if '*' in args.labels or '?' in args.labels or '[' in args.labels:
        label_paths = [Path(p) for p in glob.glob(args.labels)]
    else:
        label_paths = list(Path(args.labels).glob('*.zarr.zip'))

    assert len(label_paths) > 0, f'No label files found in {args.labels}'

    if args.imagedir is None:
        args.imagedir = args.chipdir / 'IMAGES'

    create_chips(label_paths, args.platform, dates, args.strategy, args.maxcloudpct, args.chipdir, args.imagedir)


if __name__ == '__main__':
    main()

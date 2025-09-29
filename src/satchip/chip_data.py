import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import xarray as xr
from tqdm import tqdm

import satchip
from satchip import utils
from satchip.chip_hls import get_hls_data
from satchip.chip_sentinel1rtc import get_rtc_paths_for_chips, get_s1rtc_chip_data
from satchip.chip_sentinel2 import get_s2l2a_data
from satchip.terra_mind_grid import TerraMindGrid


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


def chip_data(
    label_path: Path,
    platform: str,
    date_start: datetime,
    date_end: datetime,
    strategy: str,
    max_cloud_pct: int,
    output_dir: Path,
    scratch_dir: Path,
) -> xr.Dataset:
    labels = utils.load_chip(label_path)
    date = labels.time.data[0].astype('M8[ms]').astype(datetime)
    bounds = labels.attrs['bounds']

    grid = TerraMindGrid([bounds[1] - 1, bounds[3] + 1], [bounds[0] - 1, bounds[2] + 1])  # type: ignore
    terra_mind_chips = [c for c in grid.terra_mind_chips if c.name in list(labels.sample.data)]

    opts: utils.ChipDataOpts = {'strategy': strategy, 'date_start': date_start, 'date_end': date_end}
    if platform in ['S2L2A', 'HLS']:
        opts['max_cloud_pct'] = max_cloud_pct

    if platform == 'S1RTC':
        rtc_paths_for_chips = get_rtc_paths_for_chips(terra_mind_chips, bounds, scratch_dir, opts)

    data_chips = []
    for chip in tqdm(terra_mind_chips):
        if platform == 'S1RTC':
            rtc_paths = rtc_paths_for_chips[chip.name]
            chip_data = get_s1rtc_chip_data(chip, rtc_paths, scratch_dir, opts=opts)
        elif platform == 'S2L2A':
            chip_data = get_s2l2a_data(chip, scratch_dir, opts=opts)
        elif platform == 'HLS':
            chip_data = get_hls_data(chip, scratch_dir, opts=opts)
        else:
            raise Exception(f'Unknown platform {platform}')

        data_chips.append(chip_data)

    times = np.unique(np.concatenate([dc.time.data for dc in data_chips]))
    for i, data_chip in enumerate(data_chips):
        if len(data_chip.time) < len(times):
            data_chips[i] = fill_missing_times(data_chip, times)
    attrs = {'date_created': date.isoformat(), 'satchip_version': satchip.__version__, 'bounds': labels.attrs['bounds']}
    dataset = xr.Dataset(attrs=attrs)
    dataset['data'] = xr.combine_by_coords(data_chips, join='override')
    output_path = output_dir / (label_path.with_suffix('').with_suffix('').name + f'_{platform}.zarr.zip')
    utils.save_chip(dataset, output_path)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('labelpath', type=Path, help='Path to the label image')
    parser.add_argument('platform', choices=['S2L2A', 'S1RTC', 'HLS'], type=str, help='Dataset to create chips for')
    parser.add_argument('daterange', type=str, help='Inclusive date range to search for data in the format Ymd-Ymd')
    parser.add_argument('--maxcloudpct', default=100, type=int, help='Maximum percent cloud cover for a data chip')
    parser.add_argument('--outdir', default='.', type=Path, help='Output directory for the chips')
    parser.add_argument(
        '--scratchdir', default=None, type=Path, help='Output directory for scratch files if you want to keep them'
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

    params = (
        args.labelpath,
        args.platform,
        date_start,
        date_end,
        args.strategy,
        args.maxcloudpct,
        args.outdir,
    )

    if args.scratchdir is not None:
        chip_data(*params, args.scratchdir)
    else:
        with TemporaryDirectory() as tmp_dir:
            scratch_dir = Path(tmp_dir)
            chip_data(*params, scratch_dir)


if __name__ == '__main__':
    main()

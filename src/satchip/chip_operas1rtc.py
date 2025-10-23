from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rioxarray
import shapely
import xarray as xr
from earthaccess.results import DataGranule
from osgeo import gdal

from satchip import utils
from satchip.chip_hls import get_geometry, get_product_id
from satchip.chip_xr_base import create_dataset_chip, create_template_da
from satchip.terra_mind_grid import TerraMindChip


gdal.UseExceptions()

S1RTC_BANDS = ('VV', 'VH')


class RTCGroup:
    def __init__(self, granules: list[DataGranule]) -> None:
        self.granules = granules
        self.ids = [get_product_id(g['umm']) for g in granules]
        orbits = set([_get_orbit(g['umm']) for g in granules])
        assert len(orbits) == 1, 'RTCGroup can only contain granules from the same orbit.'
        self.orbit = orbits.pop()
        self.group_id = self.ids[0][:40]
        self.date = min([datetime.strptime(id.split('_')[4], '%Y%m%dT%H%M%SZ') for id in self.ids])
        self.footprint = shapely.unary_union([get_geometry(g['umm']) for g in granules])
        self.vv_paths = None
        self.vv_vrt = None
        self.vh_paths = None
        self.vh_vrt = None

    def download(self, image_dir: Path) -> None:
        vv_paths = []
        vh_paths = []
        for granule in self.granules:
            id = get_product_id(granule['umm'])
            files = list(image_dir.glob(f'{id}*.tif'))
            names = [f.name for f in files]
            if f'{id}_VV.tif' not in names or f'{id}_VH.tif' not in names:
                earthaccess.download([granule], image_dir, pqdm_kwargs={'disable': True})
            vv_path = image_dir / f'{id}_VV.tif'
            assert vv_path.exists(), f'Missing downloaded file: {vv_path}'
            vv_paths.append(vv_path)
            vh_path = image_dir / f'{id}_VH.tif'
            assert vh_path.exists(), f'Missing downloaded file: {vh_path}'
            vh_paths.append(vh_path)
        self.vv_paths = vv_paths
        self.vv_vrt = image_dir / f'{self.group_id}_VV.vrt'
        gdal.BuildVRT(str(self.vv_vrt), [str(p) for p in vv_paths])
        self.vh_paths = vh_paths
        self.vh_vrt = image_dir / f'{self.group_id}_VH.vrt'
        gdal.BuildVRT(str(self.vh_vrt), [str(p) for p in vh_paths])

    def confirm_download(self) -> None:
        assert self.vv_vrt is not None and self.vh_vrt is not None, 'RTCGroup images not downloaded.'


def _get_pct_intersect(rtc_group: RTCGroup, roi: shapely.geometry.Polygon) -> int:
    return int(np.round(roi.intersection(rtc_group.footprint).area / roi.area * 100))


def _get_orbit(umm: dict) -> int:
    return int(umm['OrbitCalculatedSpatialDomains'][0]['OrbitNumber'])


def group_rtcs(results: list[DataGranule]) -> list[RTCGroup]:
    orbits = set([_get_orbit(res['umm']) for res in results])
    groups = []
    for orbit in orbits:
        single_orbit_results = [res for res in results if _get_orbit(res['umm']) == orbit]
        groups.append(RTCGroup(single_orbit_results))
    return sorted(groups, key=lambda g: g.date)


def filter_to_dualpol(results: list[DataGranule]) -> list[DataGranule]:
    valid_results = []
    for res in results:
        pols = [x['Values'] for x in res['umm']['AdditionalAttributes'] if x['Name'] == 'POLARIZATION'][0]
        if 'VV' in pols and 'VH' in pols:
            valid_results.append(res)
    return valid_results


def get_scenes(groups: list[RTCGroup], roi: shapely.geometry.Polygon, strategy: str) -> list[RTCGroup]:
    intersecting = [group for group in groups if _get_pct_intersect(group, roi) > 95]
    if strategy == 'BEST':
        return intersecting[:1]
    elif strategy == 'ALL':
        return intersecting
    else:
        raise ValueError(f'Strategy must be either BEST or ALL. Got {strategy}')


def get_operartc_data(chip: TerraMindChip, image_dir: Path, opts: utils.ChipDataOpts) -> xr.Dataset:
    """Returns XArray DataArray of a OPERA S1-RTC for the given chip and selection startegy."""
    date_start = opts['date_start']
    date_end = opts['date_end'] + timedelta(days=1)  # inclusive end
    earthaccess.login()
    results = earthaccess.search_data(
        short_name=['OPERA_L2_RTC-S1_V1'], bounding_box=chip.bounds, temporal=(date_start, date_end)
    )
    results = filter_to_dualpol(results)
    rtc_groups = group_rtcs(results)
    roi = shapely.box(*chip.bounds)
    roi_buffered = roi.buffer(0.01)
    strategy = opts.get('strategy', 'BEST').upper()
    timesteps = get_scenes(rtc_groups, roi_buffered, strategy)
    assert len(timesteps) > 0, f'No OPERA RTC scenes found for chip {chip.name} between {date_start} and {date_end}.'
    [timestep.download(image_dir) for timestep in timesteps]
    template = create_template_da(chip)
    timestep_arrays = []
    for scene in timesteps:
        scene.confirm_download()
        band_arrays = []
        for band in S1RTC_BANDS:
            image_path = scene.vv_vrt if band == 'VV' else scene.vh_vrt
            da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi_buffered.bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            band_arrays.append(da_reproj.data.squeeze())
        band_array = np.stack(band_arrays, axis=0)
        timestep_arrays.append(band_array)
    data_array = np.stack(timestep_arrays, axis=0)
    dates = [scene.date for scene in timesteps]
    dataset = create_dataset_chip(data_array, chip, dates, S1RTC_BANDS)
    return dataset

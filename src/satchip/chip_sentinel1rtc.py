from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import hyp3_sdk
import numpy as np
import rioxarray
import shapely
import xarray as xr
from asf_search import S1Product
from hyp3_sdk import Job
from hyp3_sdk.util import extract_zipped_product

from satchip.chip_xr_base import create_template_da
from satchip.terra_mind_grid import TerraMindChip


def get_rtc_paths_for_chips(terra_mind_chips: list, bounds: list, scratch_dir: Path, opts: dict) -> list:
    date_start, date_end = opts['date_start'], opts['date_end']
    check_bounds_size(bounds)
    granules = get_granules(bounds, date_start, date_end)
    slcs_for_chips = pair_slcs_to_chips(terra_mind_chips, granules, opts['strategy'])
    assert len(slcs_for_chips) == len(terra_mind_chips)

    rtc_paths_for_chips = get_rtcs_for(slcs_for_chips, scratch_dir)

    return rtc_paths_for_chips


def check_bounds_size(bounds: list[int]) -> None:
    min_lon, min_lat, max_lon, max_lat = bounds
    MAX_BOUND_AREA_DEGREES = 3
    bounds_area_degrees = (max_lon - min_lon) * (max_lat - min_lat)

    err_message = f'Bounds area is to large ({bounds_area_degrees}). Must be less than {MAX_BOUND_AREA_DEGREES} degrees'
    assert bounds_area_degrees < MAX_BOUND_AREA_DEGREES, err_message


def get_granules(bounds: list[int], date_start: datetime, date_end: datetime) -> list:
    date_start = date_start
    date_end = date_end + timedelta(days=1)  # inclusive end
    roi = shapely.box(*bounds)
    search_results = asf.geo_search(
        intersectsWith=roi.wkt,
        start=date_start,
        end=date_end,
        beamMode=asf.constants.BEAMMODE.IW,
        polarization=asf.constants.POLARIZATION.VV_VH,
        platform=asf.constants.PLATFORM.SENTINEL1,
        processingLevel=asf.constants.PRODUCT_TYPE.SLC,
    )

    return search_results


def pair_slcs_to_chips(chips: list, granules: list, strategy: str) -> dict:
    slcs_for_chips = {}
    for chip in chips:
        chip_roi = shapely.box(*chip.bounds)
        intersecting = [granule for granule in granules if get_pct_intersect(granule, chip_roi) > 95]
        intersecting = sorted(intersecting, key=lambda g: (-get_pct_intersect(g, chip_roi), g.properties['startTime']))

        if len(intersecting) < 1:
            raise ValueError(f'No products found for chip {chip.name} in given date range')

        if strategy == 'BEST':
            intersecting = intersecting[:1]

        slcs_for_chips[chip.name] = intersecting

    return slcs_for_chips


def get_rtcs_for(slcs_for_chips: dict, scratch_dir: Path) -> list:
    flat_slcs = sum(slcs_for_chips.values(), [])
    slc_names = set(granule.properties['sceneName'] for granule in flat_slcs)

    hyp3 = hyp3_sdk.HyP3()
    jobs_by_scene_name = {}

    for job in hyp3.find_jobs(job_type='RTC_GAMMA'):
        if not is_valid_rtc_job(job):
            continue

        name = job.job_parameters['granules'][0]
        jobs_by_scene_name[name] = job

    hyp3_jobs = []
    for slc_name in slc_names:
        if slc_name in jobs_by_scene_name:
            job = jobs_by_scene_name[slc_name]
            hyp3_jobs.append(job)
        else:
            new_batch = hyp3.submit_rtc_job(slc_name, radiometry='gamma0', resolution=20)
            hyp3_jobs.append(list(new_batch)[0])

    batch = hyp3_sdk.Batch(hyp3_jobs)
    batch = hyp3.watch(batch)

    succeeded_jobs = [j.succeeded() for j in batch]

    assert all(succeeded_jobs), 'One or more HyP3 jobs failed'

    paths_by_slc_name = {}
    for job in batch:
        rtc_path = download_hyp3_rtc(job, scratch_dir)
        slc_name = job.job_parameters['granules'][0]

        paths_by_slc_name[slc_name] = rtc_path

    rtc_paths_for_chips = {}
    for chip_name, chip_slcs in slcs_for_chips.items():
        chip_paths = [paths_by_slc_name[name.properties['sceneName']] for name in chip_slcs]
        rtc_paths_for_chips[chip_name] = chip_paths

    return rtc_paths_for_chips


def is_valid_rtc_job(job: hyp3_sdk.Job) -> bool:
    return (
        not job.failed()
        and not job.expired()
        and job.job_parameters['radiometry'] == 'gamma0'
        and job.job_parameters['resolution'] == 20
    )


def get_pct_intersect(product: S1Product, roi: shapely.geometry.Polygon) -> int:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(np.round(100 * roi.intersection(footprint).area / roi.area))
    return intersection


def download_hyp3_rtc(job: Job, scratch_dir: Path) -> tuple[Path, Path]:
    output_path = scratch_dir / job.to_dict()['files'][0]['filename']
    output_dir = output_path.with_suffix('')
    output_zip = output_path.with_suffix('.zip')
    if not output_dir.exists():
        job.download_files(location=scratch_dir)
        extract_zipped_product(output_zip)
    vv_path = list(output_dir.glob('*_VV.tif'))[0]
    vh_path = list(output_dir.glob('*_VH.tif'))[0]
    return vv_path, vh_path


def get_s1rtc_chip_data(chip: TerraMindChip, image_sets: list[Path], scratch_dir: Path, opts: dict) -> xr.DataArray:
    roi = shapely.box(*chip.bounds)
    das = []
    template = create_template_da(chip)
    for image_set in image_sets:
        for band_name, image_path in zip(['VV', 'VH'], image_set):
            da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi.buffer(0.1).bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            da_reproj['band'] = [band_name]
            image_time = datetime.strptime(image_path.name.split('_')[2], '%Y%m%dT%H%M%S')
            da_reproj = da_reproj.expand_dims({'time': [image_time]})
            da_reproj['x'] = np.arange(0, chip.ncol)
            da_reproj['y'] = np.arange(0, chip.nrow)
            da_reproj.attrs = {}
            das.append(da_reproj)
    dataarray = xr.combine_by_coords(das, join='override').drop_vars('spatial_ref')
    assert isinstance(dataarray, xr.DataArray)
    dataarray = dataarray.expand_dims({'sample': [chip.name], 'platform': ['S1RTC']})
    return dataarray

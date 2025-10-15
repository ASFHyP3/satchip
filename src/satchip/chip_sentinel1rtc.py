from datetime import datetime, timedelta
from pathlib import Path

import asf_search as asf
import hyp3_sdk
import numpy as np
import rioxarray
import shapely
import xarray as xr

from satchip import utils
from satchip.chip_xr_base import create_dataset_chip, create_template_da
from satchip.terra_mind_grid import TerraMindChip


S1RTC_BANDS = ['VV', 'VH']


def get_rtc_paths_for_chips(
    terra_mind_chips: list[TerraMindChip], image_dir: Path, opts: utils.ChipDataOpts
) -> dict[str, list[utils.RtcImageSet]]:
    bounds = utils.get_overall_bounds([chip.bounds for chip in terra_mind_chips])
    _check_bounds_size(bounds)
    granules = _get_granules(bounds, opts['date_start'], opts['date_end'])
    slcs_for_chips = _get_slcs_for_each_chip(terra_mind_chips, granules, opts['strategy'])
    assert len(slcs_for_chips) == len(terra_mind_chips)

    rtc_paths_for_chips = _get_rtcs_for(slcs_for_chips, image_dir)
    return rtc_paths_for_chips


def _check_bounds_size(bounds: list[float]) -> None:
    min_lon, min_lat, max_lon, max_lat = bounds
    MAX_BOUND_AREA_DEGREES = 3
    bounds_area_degrees = (max_lon - min_lon) * (max_lat - min_lat)

    err_message = f'Bounds area is to large ({bounds_area_degrees}). Must be less than {MAX_BOUND_AREA_DEGREES} degrees'
    assert bounds_area_degrees < MAX_BOUND_AREA_DEGREES, err_message


def _get_granules(bounds: list[float], date_start: datetime, date_end: datetime) -> list[asf.S1Product]:
    date_start = date_start
    date_end = date_end + timedelta(days=1)  # inclusive end
    roi = shapely.box(*bounds)  # type: ignore
    search_results = asf.geo_search(
        intersectsWith=roi.wkt,
        start=date_start,
        end=date_end,
        beamMode=asf.constants.BEAMMODE.IW,
        polarization=asf.constants.POLARIZATION.VV_VH,
        platform=asf.constants.PLATFORM.SENTINEL1,
        processingLevel=asf.constants.PRODUCT_TYPE.SLC,
    )

    return list(search_results)


def _get_slcs_for_each_chip(
    chips: list[TerraMindChip], granules: list[asf.S1Product], strategy: str, intersection_pct: int = 95
) -> dict[str, list[asf.S1Product]]:
    slcs_for_chips: dict[str, list[asf.S1Product]] = {}

    for chip in chips:
        chip_roi = shapely.box(*chip.bounds)
        intersecting = [granule for granule in granules if _get_pct_intersect(granule, chip_roi) > intersection_pct]
        intersecting = sorted(intersecting, key=lambda g: (-_get_pct_intersect(g, chip_roi), g.properties['startTime']))

        if len(intersecting) < 1:
            raise ValueError(f'No products found for chip {chip.name} in given date range')

        if strategy == 'BEST':
            slcs_for_chips[chip.name] = intersecting[:1]
        else:
            slcs_for_chips[chip.name] = intersecting

    return slcs_for_chips


def _get_pct_intersect(product: asf.S1Product, roi: shapely.geometry.Polygon) -> int:
    footprint = shapely.geometry.shape(product.geometry)
    intersection = int(np.round(100 * roi.intersection(footprint).area / roi.area))
    return intersection


def _get_rtcs_for(
    slcs_for_chips: dict[str, list[asf.S1Product]], image_dir: Path
) -> dict[str, list[utils.RtcImageSet]]:
    flat_slcs = sum(slcs_for_chips.values(), [])
    slc_names = set(granule.properties['sceneName'] for granule in flat_slcs)

    finished_rtc_jobs = _process_rtcs(slc_names)

    paths_for_slc_name: dict[str, utils.RtcImageSet] = {}
    for job in finished_rtc_jobs:
        rtc_image_set = _download_hyp3_rtc(job, image_dir)
        slc_name = job.job_parameters['granules'][0]  # type: ignore
        paths_for_slc_name[slc_name] = rtc_image_set

    rtc_paths_for_chips: dict[str, list[utils.RtcImageSet]] = {}
    for chip_name, chip_slcs in slcs_for_chips.items():
        image_sets = [paths_for_slc_name[name.properties['sceneName']] for name in chip_slcs]
        rtc_paths_for_chips[chip_name] = image_sets

    return rtc_paths_for_chips


def _process_rtcs(slc_names: set[str]) -> hyp3_sdk.Batch:
    hyp3 = hyp3_sdk.HyP3()
    jobs_by_scene_name = _get_rtc_jobs_by_scene_name(hyp3)

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
    assert all([j.succeeded() for j in batch]), 'One or more HyP3 jobs failed'  # type: ignore

    return batch  # type: ignore


def _get_rtc_jobs_by_scene_name(hyp3: hyp3_sdk.HyP3) -> dict[str, hyp3_sdk.Job]:
    jobs_by_scene_name = {}

    for job in hyp3.find_jobs(job_type='RTC_GAMMA'):
        if not _is_valid_rtc_job(job):
            continue

        name = job.job_parameters['granules'][0]  # type: ignore
        jobs_by_scene_name[name] = job

    return jobs_by_scene_name


def _is_valid_rtc_job(job: hyp3_sdk.Job) -> bool:
    return (
        not job.failed()
        and not job.expired()
        and job.job_parameters['radiometry'] == 'gamma0'  # type: ignore
        and job.job_parameters['resolution'] == 20  # type: ignore
    )


def _download_hyp3_rtc(job: hyp3_sdk.Job, image_dir: Path) -> utils.RtcImageSet:
    output_path = image_dir / job.to_dict()['files'][0]['filename']
    output_dir = output_path.with_suffix('')
    output_zip = output_path.with_suffix('.zip')
    if not output_dir.exists():
        job.download_files(location=image_dir)
        hyp3_sdk.util.extract_zipped_product(output_zip)
    vv_path = list(output_dir.glob('*_VV.tif'))[0]
    vh_path = list(output_dir.glob('*_VH.tif'))[0]
    image_set = utils.RtcImageSet(vv=vv_path, vh=vh_path)
    return image_set


def get_s1rtc_chip_data(chip: TerraMindChip, image_sets: list[utils.RtcImageSet]) -> xr.Dataset:
    roi = shapely.box(*chip.bounds)
    template = create_template_da(chip)
    timestep_arrays = []
    for image_set in image_sets:
        band_arrays = []
        for band in S1RTC_BANDS:
            image_path = image_set._asdict()[band]
            da = rioxarray.open_rasterio(image_path).rio.clip_box(*roi.buffer(0.1).bounds, crs='EPSG:4326')  # type: ignore
            da_reproj = da.rio.reproject_match(template)
            band_arrays.append(da_reproj.data.squeeze())
        band_array = np.stack(band_arrays, axis=0)
        timestep_arrays.append(band_array)
    data_array = np.stack(timestep_arrays, axis=0)
    dates = [datetime.strptime(image_set[0].name.split('_')[2], '%Y%m%dT%H%M%S') for image_set in image_sets]
    dataset = create_dataset_chip(data_array, chip, dates, S1RTC_BANDS)
    return dataset

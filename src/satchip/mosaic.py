import datetime
import shutil
from pathlib import Path

import earthaccess
import hls
import numpy as np
import opera_rtc
import rasterio
from modality import Modality
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds


def data_over_swath(swath, modalities: list[Modality], output_path: Path):
    data_paths = {
        'RAW': output_path / 'RAW',
        'REPROJECTED': output_path / 'REPROJECTED',
        'MOSAIC': output_path / 'MOSAIC',
    }

    for p in ('MOSAIC',):
        shutil.rmtree(data_paths[p], ignore_errors=True)

    for p in data_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    swathID = f'{int(swath["swathID"]):04d}'

    stacked = {}
    for modality in modalities:
        print(f'Localizing data for {modality.id}.')

        bounding_box = swath['buffered_event_background'].bounds

        start_date = swath['ls5hlsDate'] if modality.id == 'HLS' else swath['s1Date']

        results = search_data(bounding_box, start_date, modality)

        local_files = earthaccess.download(results, local_path=data_paths['RAW'], show_progress=True)

        data_tifs = [f for f in local_files if f.name.endswith('.tif')]
        reprojected_tifs = _reproject_files(data_tifs, output_path=data_paths['REPROJECTED'])

        if len(data_tifs) == 0:
            print(f'Skipping: no data for swath {swathID}')
            continue

        mod_merged = {}

        for band in modality.all_bands:
            band_files = [f for f in reprojected_tifs if band in band_from_filename(f.name, modality)]
            merged_name = make_merge_name(swathID, start_date, band, modality)

            merged_band_path = _merge(band_files, output_file=data_paths['MOSAIC'] / merged_name)

            mod_merged[band] = merged_band_path

        print(f'Stacking bands for {modality.id}')
        stacked_data = _stack_bands(mod_merged, data_bands=modality.stack_bands, stacked_name='BANDS')
        stacked[modality.id] = stacked_data

    print(f'Warp band {band} data to same area')
    warped = _warp_over_swath(
        data=stacked,
        bounding_box_4326=bounding_box,
        output_dir=output_path,
    )

    return warped


def search_data(bounding_box: tuple, start_date: datetime.datetime, modality: Modality):
    results = {}

    if modality.id == 'HLS':
        results = hls.search_hls_data(start_date=start_date, bounding_box=bounding_box)
    elif modality.id == 'RTC':
        results = opera_rtc.search_rtc_data(start_date=start_date, bounding_box=bounding_box)

    return results


def band_from_filename(filename, modality):
    if modality.id == 'HLS':
        band = hls.band_from_hls_filename(filename)
    elif modality.id == 'RTC':
        band = opera_rtc.band_from_rtc_filename(filename)

    return band


def make_merge_name(swathID: str, start_date: datetime.datetime, band: str, modality: Modality):
    date_str = start_date.date().isoformat()

    return f'{swathID}.{modality.id}.{date_str}.{band}.tif'


def _reproject_files(files: list[Path], output_path: Path) -> list[Path]:
    reprojected_paths = [output_path / f'{granule.name}' for granule in files]

    for granule, output_path in zip(files, reprojected_paths):
        if output_path.exists():
            continue

        print(f'reprojecting to wgs84: {output_path.name}')
        _reproject_file(granule, output_path)

    return reprojected_paths


def _reproject_file(local_file: Path, reprojected_file: Path, epsg=4326) -> None:
    # https://rasterio.readthedocs.io/en/stable/topics/reproject.html#reprojecting-a-geotiff-dataset
    with rasterio.open(local_file) as src:
        dst_crs = CRS.from_epsg(epsg)
        transform, width, height = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)

        dst_kwargs = src.meta.copy()
        dst_kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})

        with rasterio.open(reprojected_file, 'w', **dst_kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                )


def _merge(band_files: list[Path], output_file: Path) -> Path:
    band_datasets = [rasterio.open(rtc_tif) for rtc_tif in band_files]

    master_crs = band_datasets[0].crs
    for ds in band_datasets[1:]:
        if ds.crs != master_crs:
            ds.crs = master_crs

    try:
        mosaic, out_trans = merge(band_datasets)
        mosaic = np.squeeze(mosaic)

        out_meta = band_datasets[0].meta.copy()

        out_meta.update(
            {
                'driver': 'GTiff',
                'height': mosaic.shape[0],
                'width': mosaic.shape[1],
                'transform': out_trans,
                'crs': band_datasets[0].crs,
            }
        )

        with rasterio.open(output_file, 'w', **out_meta) as dst:
            dst.write(mosaic, 1)
    finally:
        for ds in band_datasets:
            ds.close()

    return output_file


def _stack_bands(merged: dict[str, Path], data_bands: tuple[str], stacked_name: str) -> None:
    with rasterio.open(merged[data_bands[0]]) as src:
        meta = src.meta.copy()

    band = data_bands[0]
    meta.update(count=len(data_bands), dtype=np.float32)
    stacked_file_name = _rename(merged[band], f'{band}.tif', f'{stacked_name}.tif')

    with rasterio.open(stacked_file_name, 'w', **meta) as dst:
        for idx, band in enumerate(data_bands, start=1):
            with rasterio.open(merged[band]) as src:
                dst.write(src.read(1), idx)

    merged[stacked_name] = stacked_file_name
    return merged


def _rename(path: Path, extension: str, mask_name: str) -> Path:
    return path.parent / path.name.replace(extension, mask_name)


def _warp_over_swath(data, bounding_box_4326, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = next(iter(next(iter(data.values())).values()))
    dst_transform, width, height, dst_crs = _build_common_grid(bounding_box_4326, reference_path)

    output = {}
    for sensor, bands in data.items():
        output[sensor] = {}
        for band_name, input_path in bands.items():
            out_path = output_dir / input_path.name
            _warp_single(input_path, out_path, dst_transform, width, height, dst_crs, band_name)
            output[sensor][band_name] = out_path

    return output


def _warp_single(input_path, output_path, dst_transform, width, height, dst_crs, band_name):
    CATEGORICAL_BANDS = {'Fmask', 'mask'}
    resampling = Resampling.nearest if band_name in CATEGORICAL_BANDS else Resampling.bilinear

    with rasterio.open(input_path) as src:
        dst_data = np.zeros((src.count, height, width), dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, list(range(1, src.count + 1))),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
            dst_nodata=src.nodata,
        )

        out_meta = src.meta.copy()
        out_meta.update(
            {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'transform': dst_transform,
                'crs': dst_crs,
            }
        )

        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(dst_data)

    return output_path


def _build_common_grid(bounding_box_4326, reference_path):
    dst_crs = CRS.from_epsg(4326)
    minx, miny, maxx, maxy = bounding_box_4326

    with rasterio.open(reference_path) as ref:
        bounds_4326 = transform_bounds(ref.crs, dst_crs, *ref.bounds, densify_pts=21)
        ref_width = ref.width
        ref_height = ref.height

    ref_bbox_width = bounds_4326[2] - bounds_4326[0]
    ref_bbox_height = bounds_4326[3] - bounds_4326[1]
    res_x = ref_bbox_width / ref_width
    res_y = ref_bbox_height / ref_height

    minx = np.floor(minx / res_x) * res_x
    miny = np.floor(miny / res_y) * res_y
    maxx = np.ceil(maxx / res_x) * res_x
    maxy = np.ceil(maxy / res_y) * res_y

    width = int(round((maxx - minx) / res_x))
    height = int(round((maxy - miny) / res_y))
    dst_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    return dst_transform, width, height, dst_crs

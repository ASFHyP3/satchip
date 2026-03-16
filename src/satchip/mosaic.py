from pathlib import Path
import datetime

import earthaccess
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.mask import mask
from shapely.geometry import box

from modality import Modality
import hls
import opera_rtc


def data_over_swath(swath, modalities: list[Modality], output_path: Path):
    data_paths = {
        "RAW": output_path / "RAW",
        "REPROJECTED": output_path / "REPROJECTED",
    }

    for item in output_path.glob('*.tif'):
        if item.is_dir():
            continue

        item.unlink()

    for p in data_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    swathID = f"{int(swath['swathID']):04d}"

    merged = {}

    for modality in modalities:
        print(f'Localizing data for {modality.id}.')

        bounding_box = swath["buffered_event_background"].bounds
        # transformer = Transformer.from_crs(32615, 4326, always_xy=True)
        # minx, miny = transformer.transform(minx, miny)
        # maxx, maxy = transformer.transform(maxx, maxy)

        start_date = swath["ls5hlsDate"] if modality.id == 'HLS' else swath["s1Date"]
        print(start_date)

        results = search_data(bounding_box, start_date, modality)

        local_files = earthaccess.download(
            results, local_path=data_paths["RAW"], show_progress=True
        )

        data_tifs = [f for f in local_files if f.name.endswith(".tif")]
        reprojected_tifs = _reproject_files(data_tifs, output_path=data_paths["REPROJECTED"])

        if len(data_tifs) == 0:
            print(f"Skipping: no data for swath {swathID}")
            continue

        mod_merged = {}

        for band in modality.all_bands:
            band_files = [f for f in reprojected_tifs if band in band_from_filename(f.name, modality)]
            merged_name = make_merge_name(swathID, start_date, band, modality)

            merged_band_path = _merge(
                band_files, output_file=output_path / merged_name
            )

            print(f'Clipping band {band} data to same area')
            clipped_band_path = _clip_over_swath(merged_band_path, swath)
            mod_merged[band] = clipped_band_path

        print(f'Stacking bands for {modality.id}')
        stacked_data = _stack_bands(mod_merged, data_bands=modality.stack_bands)

        merged[modality.id] = stacked_data

    breakpoint()
    return merged


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


def _reproject_files(
    files: list[Path], output_path: Path
) -> list[Path]:

    reprojected_paths = [
        output_path / f"{granule.name}" for granule in files
    ]

    for granule, output_path in zip(files, reprojected_paths):
        if output_path.exists():
            continue

        print(f"reprojecting to wgs84: {output_path.name}")
        _reproject_file(granule, output_path)

    return reprojected_paths


def _reproject_file(local_file: Path, reprojected_file: Path, epsg=4326) -> None:
    # https://rasterio.readthedocs.io/en/stable/topics/reproject.html#reprojecting-a-geotiff-dataset
    with rasterio.open(local_file) as src:
        dst_crs = CRS.from_epsg(epsg)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        dst_kwargs = src.meta.copy()
        dst_kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )

        with rasterio.open(reprojected_file, "w", **dst_kwargs) as dst:
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
                "driver": "GTiff",
                "height": mosaic.shape[0],
                "width": mosaic.shape[1],
                "transform": out_trans,
                "crs": band_datasets[0].crs,
            }
        )

        with rasterio.open(output_file, "w", **out_meta) as dst:
            dst.write(mosaic, 1)
    finally:
        for ds in band_datasets:
            ds.close()

    return output_file


def _stack_bands(merged: dict[str, Path], data_bands: tuple[str]) -> None:
    with rasterio.open(merged[data_bands[0]]) as src:
        meta = src.meta.copy()

    band = data_bands[0]
    meta.update(count=len(data_bands), dtype=np.float32)
    stacked_file_name = _rename(merged[band], f"{band}.tif", "BANDS.tif")

    with rasterio.open(stacked_file_name, "w", **meta) as dst:
        for idx, band in enumerate(data_bands, start=1):
            with rasterio.open(merged[band]) as src:

                dst.write(src.read(1), idx)

    merged["BANDS"] = stacked_file_name
    return merged


def _rename(path: Path, extension: str, mask_name: str) -> Path:
    return path.parent / path.name.replace(extension, mask_name)


def _clip_over_swath(input_path: Path, swath) -> Path:
    bounding_box = swath["buffered_event_background"].bounds

    with rasterio.open(input_path) as src:
        out_image, out_transform = mask(src, shapes=[box(*bounding_box)], crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "crs": src.crs
        })

    with rasterio.open(input_path, "w", **out_meta) as dest:
        dest.write(out_image)

    return input_path

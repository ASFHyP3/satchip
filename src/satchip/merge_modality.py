import datetime
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, transform_bounds
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge
from rasterio.transform import from_bounds, Affine

from satchip import models


def merge_modality(
    modality_files: list[Path],
    modality: models.Modality,
    event: models.Event,
    output_path: Path,
    selected_bands: list[models.Band] | None = None
) -> list[Path]:
    output_path.mkdir(exist_ok=True, parents=True)

    if len(modality_files) == 0:
        print(f"Warning: no data for {event.name}")
        return []

    if selected_bands is None:
        selected_bands = modality['bands']

    merged = []

    for band in selected_bands:
        band_files = [f for f in modality_files if band.id in models.band_id_from_filename(f.name, modality['id'])]

        merged_name = _make_merge_name(event.name, event.date, band.shortname, modality['id'])

        merged_band_path = _merge(
            band_files, output_file=output_path / merged_name
        )

        merged.append(merged_band_path)

    return merged


def _make_merge_name(event_name: str, start_date: datetime.datetime, band: str, modality_id: str):
    date_str = start_date.date().isoformat()

    return f'{event_name}.{modality_id}.{date_str}.{band}.tif'


def _merge(band_files: list[Path], output_file: Path) -> Path:
    band_datasets = [rasterio.open(band_file) for band_file in band_files]

    reference_crs = band_datasets[0].crs
    for ds in band_datasets[1:]:
        if ds.crs != reference_crs:
            ds.crs = reference_crs

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


def stack_bands(band_files: Iterable[Path], stacked_filename: Path) -> Path:
    stacked_filename.parent.mkdir(exist_ok=True, parents=True)

    with rasterio.open(band_files[0]) as src:
        meta = src.meta.copy()

    meta.update(count=len(band_files), dtype=np.float32)

    with rasterio.open(stacked_filename, "w", **meta) as dst:
        for idx, band_file in enumerate(band_files, start=1):
            with rasterio.open(band_file) as src:

                dst.write(src.read(1), idx)

    return stacked_filename


def reproject_files(
    files: list[Path], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(exist_ok=True, parents=True)

    reprojected_paths = [
        output_dir / f"{file.name}" for file in files
    ]

    for file, output in zip(files, reprojected_paths):
        if output.exists():
            continue

        print(f"reprojecting to wgs84: {output.name}")
        reproject_file(file, output)

    return reprojected_paths


def reproject_file(local_file: Path, reprojected_file: Path, epsg=4326) -> None:
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


def warp_to_reference(
    reference_path: Path,
    data_files: Iterable[Path],
    output_dir: Path,
    bounding_box_wgs84: tuple(float, float, float, float)
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    dst_transform, width, height, dst_crs = _build_common_grid(
        bounding_box_wgs84, reference_path
    )

    files = (reference_path, *data_files)

    output = []
    for data_file in files:
        out_path = output_dir / data_file.name

        _warp_single(
            data_file, out_path,
            dst_transform, width, height, dst_crs,
        )

        output.append(out_path)

    return output


def _warp_single(input_path: Path, output_path: Path, dst_transform: Affine, width: int, height: int, dst_crs: CRS) -> Path:
    resampling = _get_resampling_method(input_path)

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
        out_meta.update({
            "driver":    "GTiff",
            "height":    height,
            "width":     width,
            "transform": dst_transform,
            "crs":       dst_crs,
        })

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(dst_data)

    return output_path


def _get_resampling_method(filepath: Path) -> Resampling:
    filename = filepath.name.lower()

    if "fmask" in filename or "mask" in filename:
        return Resampling.nearest
    else:
        return Resampling.bilinear



def _build_common_grid(bounding_box_4326: tuple(float, float, float, float), reference_path: Path) -> tuple[Affine, int, int, CRS]:
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

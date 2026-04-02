import datetime
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import calculate_default_transform, reproject
from rasterio.merge import merge

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

def _rename(path: Path, extension: str, mask_name: str) -> Path:
    return path.parent / path.name.replace(extension, mask_name)

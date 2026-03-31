from pathlib import Path
import datetime

import numpy as np
import rasterio
from rasterio.merge import merge

from satchip import models


def merge_modality(modality_files: list[Path], modality: models.Modality, event: models.Event, output_path: Path, selected_bands: list[models.Band] | None = None):
    output_path.mkdir(exist_ok=True, parents=True)

    if len(modality_files) == 0:
        print(f"Warning: no data for {event.name}")
        return []

    merged = {}

    if selected_bands is None:
        selected_bands = modality['bands']

    for band in selected_bands:
        band_files = [f for f in modality_files if band.id in models.band_id_from_filename(f.name, modality['id'])]

        merged_name = _make_merge_name(event.name, event.date, band.shortname, modality['id'])

        merged_band_path = _merge(
            band_files, output_file=output_path / merged_name
        )

        merged[band] = merged_band_path

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

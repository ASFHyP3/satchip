from pathlib import Path

import rasterio
from rasterio.merge import merge
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from shapely.geometry import box

from satchip import models, merge_modality


def view_merged(
    stacked_data_file: Path,
    event: models.Event,
    modality: models.Modality,
    rgb_bands: tuple[int, int, int],
    save_to_file: Path | None = None,
    quite: bool = False,
):
    crs_pc = ccrs.PlateCarree()

    with rasterio.open(stacked_data_file) as ds:
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        band_data = ds.read()

    img = get_img(band_data, modality, rgb_bands)

    # plot BANDS and geom
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw={'projection': crs_pc},
        figsize=(12, 12),
        layout='constrained',
    )

    event_geom = event.wgs84_geometry

    ax.imshow(img, extent=full_extent, origin='upper', transform=crs_pc)
    ax.add_geometries([event_geom], edgecolor='red', linewidth=2, facecolor='none', crs=crs_pc)

    if save_to_file:
        plt.savefig(
            save_to_file,
            dpi=300,
            bbox_inches='tight',
        )

    if not quite:
        plt.show()

    plt.close(fig)


def view_chip(
    chip: models.ChipStack,
    modality: models.Modality,
    rgb_bands: tuple[int, int, int],
    save_to_file: Path | None = None,
    quite: bool = False,
):
    with rasterio.open(chip.data) as ds:
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        band_data = ds.read()

    with rasterio.open(chip.label) as ds:
        label_data = ds.read().squeeze()

    img = get_img(band_data, modality, rgb_bands)

    crs_pc = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1,
        2,
        subplot_kw={'projection': crs_pc},
        figsize=(12, 12),
        layout='constrained',
    )

    ax[0].imshow(img, extent=full_extent, origin='upper', transform=crs_pc)
    ax[1].imshow(label_data, extent=full_extent, origin='upper', transform=crs_pc)

    if save_to_file:
        plt.savefig(
            save_to_file,
            dpi=300,
            bbox_inches='tight',
        )

    if not quite:
        plt.show()

    plt.close(fig)


def view_chips(
    chips: models.ChipStack,
    modality: models.Modality,
    rgb_bands: tuple[int, int, int],
    save_to_file: Path | None = None,
    quite: bool = False,
):
    merged_data = _merge_chips([chip.data for chip in chips], Path.cwd() / 'merged_data.tif')
    merged_label = _merge_chips([chip.label for chip in chips], Path.cwd() / 'merged_label.tif')
    try:
        with rasterio.open(merged_data) as ds:
            bounds = ds.bounds
            full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            band_data = ds.read()

        with rasterio.open(merged_label) as ds:
            label_data = ds.read().squeeze()

        img = get_img(band_data, modality, rgb_bands)

        crs_pc = ccrs.PlateCarree()
        # plot BANDS and geom
        fig, ax = plt.subplots(
            1,
            2,
            subplot_kw={'projection': crs_pc},
            figsize=(12, 12),
            layout='constrained',
        )

        ax[0].imshow(img, extent=full_extent, origin='upper', transform=crs_pc)
        ax[1].imshow(label_data, extent=full_extent, origin='upper', transform=crs_pc)

        if save_to_file:
            plt.savefig(
                save_to_file,
                dpi=300,
                bbox_inches='tight',
            )

        if not quite:
            plt.show()

        plt.close(fig)
    finally:
        merged_data.unlink(missing_ok=True)
        merged_label.unlink(missing_ok=True)


def _merge_chips(
    chips: list[Path],
    output_file: Path,
) -> Path:
    datasets = [rasterio.open(p) for p in chips]

    try:
        mosaic, transform = merge(datasets)

        out_meta = datasets[0].meta.copy()
        out_meta.update(
            {
                'height': mosaic.shape[1],
                'width': mosaic.shape[2],
                'transform': transform,
            }
        )

        with rasterio.open(output_file, 'w', **out_meta) as dst:
            if len(mosaic.shape) == 2:
                dst.write(mosaic, 1)
            else:
                dst.write(mosaic)
    finally:
        for ds in datasets:
            ds.close()

    return output_file


def get_img(band_data: np.ndarray, modality: models.Modality, rgb_bands: tuple[int, int, int]):
    if 'HLS' in modality['id']:
        img = get_hls_img(band_data, rgb_bands)
    elif 'RTC' in modality['id']:
        img = get_rtc_img(band_data, rgb_bands)

    return img


def get_hls_img(hls_data: np.ndarray, rgb_bands: tuple[int, int, int]) -> np.ndarray:
    r_band, g_band, b_band = rgb_bands

    r = bytescale(np.sqrt(np.clip(hls_data[r_band] / 10000.0, 0, 2)), 0, 0.5)
    g = bytescale(np.sqrt(np.clip(hls_data[g_band] / 10000.0, 0, 2)), 0, 0.5)
    b = bytescale(np.sqrt(np.clip(hls_data[b_band] / 10000.0, 0, 2)), 0, 0.5)

    rgb = np.dstack((r, g, b))
    return rgb


def bytescale(arr, cmin=0, cmax=1, low=0, high=255):
    # clip the data to be in the range of cmin to cmax
    arr = np.clip(arr, cmin, cmax)
    high = float(high)
    low = float(low)
    cmax = float(cmax)
    cmin = float(cmin)
    m = (high - low) / (cmax - cmin)  # slope
    b = high - (m * cmax)  # intercept
    arr = np.uint8((m * arr) + b)
    return arr


def get_rtc_img(rtc_data: np.ndarray, rgb_bands: tuple[int, int, int]) -> np.ndarray:
    r_band, g_band, b_band = rgb_bands

    r = normalize_image_array(np.sqrt(rtc_data[r_band]), 0.14, 0.52)
    g = normalize_image_array(np.sqrt(rtc_data[g_band]), 0.05, 0.259)
    b = normalize_image_array(np.sqrt(rtc_data[b_band]), 0.14, 0.52)

    img = np.stack([r, g, b], axis=-1)

    return img


def normalize_image_array(input_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    input_array = input_array.astype(float)
    scaled_array = (input_array - vmin) / (vmax - vmin)
    scaled_array[np.isnan(input_array)] = 0
    normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)

    return normalized_array

from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rasterio
from earthaccess.results import DataGranule


def search_rtc_data(start_date: datetime, bounding_box: tuple[float, float, float, float]) -> list[DataGranule]:
    final_date = start_date + timedelta(days=1)

    results = earthaccess.search_data(
        short_name=['OPERA_L2_RTC-S1_V1'],
        temporal=(start_date.strftime('%Y-%m-%d'), final_date.strftime('%Y-%m-%d')),
        bounding_box=bounding_box,
    )

    return results


def band_from_rtc_filename(filename):
    # OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif
    return filename.split('_')[-1].split('.')[0]


def make_merged_rtc_name(template_filename: str) -> str:
    """
    https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/#naming-convention
    swathID.OPERA_L2_RTC-S1_[BurstID]_[StartDateTime]_[ProductGenerationDateTime] _[Sensor]_[PixelSpacing]_[ProductVersion]_[LayerName].Ext

    Input:   1442.OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif
    Returns: 1442.OPERA_L2_RTC-133415-IW2_20170620_S1A_30_v1.0_VV.tif
    """

    # ['1442.OPERA', 'L2', 'RTC-S1', 'T063-133415-IW2', '20170620T001327Z', '20250925T045340Z', 'S1A', '30', 'v1.0', 'VV.tif']
    name_parts = template_filename.split('_')

    name_parts.pop(5)  # Remove Product Generation Time
    name_parts.pop(3)  # Remove Burst ID

    return '_'.join(name_parts)


def is_valid_rtc(mask_path: Path, label_path: Path) -> bool:
    with rasterio.open(mask_path) as ds:
        validity_mask = ds.read(1)

    with rasterio.open(label_path) as ds:
        event_mask = ds.read(1)

    is_event_pixel = event_mask == 1
    # https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/#validity-mask
    is_valid_pixel = np.isin(validity_mask, [0, 1])

    total_event_pixels = is_event_pixel.sum()
    valid_event_pixels = (is_event_pixel & is_valid_pixel).sum()

    pct_valid_data = 100.0 * valid_event_pixels / total_event_pixels
    print(f'Percent of the event with valid data: {pct_valid_data:.1f}%')

    return pct_valid_data > 50.0


def filter_rtc_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip['BANDS']) as ds:
            rtc_data = ds.read()

        with rasterio.open(chip['EVENT']) as ds:
            event_mask = ds.read(1)

        has_nan_pixels = np.isnan(rtc_data).sum() > 0

        num_pixels = event_mask.size
        num_event_pixels = np.count_nonzero(event_mask > 0)

        pct_pixels_over_event = 100.0 * (num_event_pixels / num_pixels)
        data_overlaps_event = pct_pixels_over_event > 1

        if not has_nan_pixels and data_overlaps_event:
            good_chips.append(chip)

    return good_chips


def normalize_image_array(input_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    input_array = input_array.astype(float)
    scaled_array = (input_array - vmin) / (vmax - vmin)
    scaled_array[np.isnan(input_array)] = 0
    normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)

    return normalized_array


def get_rtc_img(rtc_data: np.ndarray) -> np.ndarray:
    vv = normalize_image_array(np.sqrt(rtc_data[0]), 0.14, 0.52)
    vh = normalize_image_array(np.sqrt(rtc_data[1]), 0.05, 0.259)

    img = np.stack([vv, vh, vv], axis=-1)

    return img

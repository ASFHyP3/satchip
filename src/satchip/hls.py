from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import numpy as np
import rasterio
from earthaccess.results import DataGranule


def search_hls_data(start_date: datetime, bounding_box: tuple[float, float, float, float]) -> list[DataGranule]:
    final_date = start_date + timedelta(days=1)

    #  collection_ids = ["C2021957295-LPCLOUD"]  # S2
    collection_ids = ['C2021957657-LPCLOUD', 'C2021957295-LPCLOUD']  # S2, L30

    results = earthaccess.search_data(
        concept_id=collection_ids,
        temporal=(start_date.strftime('%Y-%m-%d'), final_date.strftime('%Y-%m-%d')),
        bounding_box=bounding_box,
        cloud_hosted=True,
    )

    return results


def band_from_hls_filename(filename):
    # HLS.L30.T15TUG.2017167T165321.v2.0.B06.tif
    parts = filename.split('.')

    sensor, band = parts[1], parts[-2]

    bands = {
        'L30': {
            'B02': 'B',
            'B03': 'G',
            'B04': 'R',
            'B05': 'N',
            'B06': 'SW1',
            'B07': 'SW2',
            'Fmask': 'Fmask',
        },
        'S30': {
            'B02': 'B',
            'B03': 'G',
            'B04': 'R',
            'B08': 'N',
            'B11': 'SW1',
            'B12': 'SW2',
            'Fmask': 'Fmask',
        },
    }

    try:
        return bands[sensor][band]
    except KeyError:
        return ''


def make_merged_hls_name(template_filename: str) -> str:
    parts = template_filename.split('.')
    parts[4] = parts[4][0:7]
    parts.pop(3)
    parts[-2] = band_from_hls_filename(template_filename)
    f_template_merge = '.'.join(parts)
    return f_template_merge


def clear_px_Fmask(Fmask: np.ndarray) -> np.ndarray:
    fmask_clear = np.array(
        [
            0,
            4,
            16,
            20,
            32,
            36,
            48,
            52,
            64,
            68,
            80,
            84,
            96,
            100,
            112,
            116,
            128,
            132,
            144,
            148,
            160,
            164,
            176,
            180,
            192,
            196,
            208,
            212,
            224,
            228,
            240,
            244,
        ],
        dtype=Fmask.dtype,
    )

    cloudmask = np.ones_like(Fmask, dtype=np.uint8)
    cloudmask[np.isin(Fmask, fmask_clear)] = 0
    cloudmask[Fmask == 255] = 255

    return cloudmask


def is_valid_hls(fmask_path: Path, event_path: Path):
    with rasterio.open(fmask_path) as ds:
        qc = clear_px_Fmask(ds.read(1))
        qc_profile = ds.profile

    with rasterio.open(event_path) as ds:
        event_mask = ds.read(1)
        event_profile = ds.profile

    print(qc_profile, event_profile)

    ny, nx = np.shape(qc)
    mask = np.zeros((ny, nx), 'uint8')

    ok = np.where((event_mask == 2) & (qc == 0))
    n_cf_event = len(ok[0])
    mask[ok] = 1

    ok = np.where((event_mask == 1) & (qc != 255))
    n_valid_event = len(ok[0])

    ok = np.where(event_mask == 1)
    n_event = len(ok[0])

    pct_cf_event = 0
    if n_valid_event == 0:
        print('No coverage')
    else:
        pct_cf_event = 100.0 * (n_cf_event / n_event)
    print('Percent CF/valid in Event:', pct_cf_event)

    return pct_cf_event > 50


def filter_hls_chips(chips: dict[str, dict]) -> list[dict]:
    good_chips = []

    for tile_id, chip in chips.items():
        with rasterio.open(chip['Fmask']) as ds:
            qc = clear_px_Fmask(ds.read(1))

        with rasterio.open(chip['EVENT']) as ds:
            event = ds.read(1)

        ny, nx = qc.shape
        n_px = 1.0 * ny * nx

        # cloud-free pixels (0 clear, 1 cloud, 255 nodata)
        n_cf = len(np.where(qc == 0)[0])
        pct_cf = 100.0 * (n_cf / n_px)

        # event pixels
        n_ev = len(np.where(event > 0)[0])

        # pct of chip in event
        pct_ev = 100.0 * (n_ev / n_px) if n_ev > 0 else 0

        if pct_cf > 95 and pct_ev > 1:
            good_chips.append(chip)

    return good_chips


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


def get_hls_img(hls_data: np.ndarray) -> np.ndarray:
    # B04
    r = bytescale(np.sqrt(np.clip(hls_data[2] / 10000.0, 0, 2)), 0, 0.5)

    # B03
    g = bytescale(np.sqrt(np.clip(hls_data[1] / 10000.0, 0, 2)), 0, 0.5)

    # B02
    b = bytescale(np.sqrt(np.clip(hls_data[0] / 10000.0, 0, 2)), 0, 0.5)

    rgb = np.dstack((r, g, b))
    return rgb

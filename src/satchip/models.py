from datetime import datetime
from dataclasses import dataclass
from typing import TypedDict, NamedTuple
from typing import Tuple

import shapely


class Band(NamedTuple):
    id: str
    name: str
    shortname: str


class Modality(TypedDict):
    id: str
    bands: Tuple[Band, ...]
    collection: str


@dataclass(frozen=True)
class Event:
    name: str
    date: datetime
    wgs84_geometry: shapely.geometry.Polygon


class ModalityError(Exception):
    pass


def bands_by_shortname(bands: tuple[Band, ...]) -> dict[str, Band]:
    return {band.shortname: band for band in bands}


def bands_by_id(bands: tuple[Band, ...]) -> dict[str, Band]:
    return {band.id: band for band in bands}


MODALITIES: dict[str, Modality] = {
    "OPERA_RTC": {
        "id": "OPERA_RTC",
        "collection": "OPERA_L2_RTC-S1_V1",
        "bands": (
            Band("VV", "VV", "VV"),
            Band("VH", "VH", "VH"),
            Band("mask", "Validitiy Mask", "mask"),
        )
    },
    "HLS_S30": {
        "id": "HLS_S30",
        "collection": "HLSS30",
        "bands": (
            Band("B02", "Blue", "B"),
            Band("B03", "Green", "G"),
            Band("B04", "Red", "R"),
            Band("B8A", "NIR Narrow", "N"),
            Band("B11", "SWIR 1", "SW1"),
            Band("B12", "SWIR 2", "SW2"),
            Band("Fmask", "Cloud Mask", "Fmask"),
        )
    },
    "HLS_L30": {
        "id": "HLS_L30",
        "collection": "HLSL30",
        "bands": (
            Band("B02", "Blue", "B"),
            Band("B03", "Green", "G"),
            Band("B04", "Red", "R"),
            Band("B05", "NIR Narrow", "N"),
            Band("B06", "SWIR 1", "SW1"),
            Band("B07", "SWIR 2", "SW2"),
            Band("Fmask", "Cloud Mask", "fmask"),
        )
    }
}

MODALITY_IDS = list(MODALITIES.keys())

# https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/
OPERA_RTC = MODALITIES['OPERA_RTC']
OPERA_RTC_BANDS = bands_by_shortname(MODALITIES['OPERA_RTC']['bands'])

# https://www.earthdata.nasa.gov/data/projects/hls/spectral-bands
HLS_S30 = MODALITIES['HLS_S30']
HLS_S30_BANDS = bands_by_shortname(MODALITIES['HLS_S30']['bands'])

HLS_L30 = MODALITIES['HLS_L30']
HLS_L30_BANDS = bands_by_shortname(MODALITIES['HLS_L30']['bands'])


def band_id_from_filename(filename: str, modality_id: str) -> Band | None:
    if 'HLS_L30' in modality_id:
        sensor, band = band_from_hls_filename(filename)
    elif 'HLS_S30' in modality_id:
        sensor, band = band_from_hls_filename(filename)
    elif 'OPERA_RTC' in modality_id:
        sensor, band = band_from_rtc_filename(filename)
    else:
        raise ModalityError(f'Modality not found {modality_id}, must be ({MODALITY_IDS})')

    if sensor not in modality_id:
        return ''

    bands = bands_by_id(MODALITIES[modality_id]['bands'])

    try:
        return bands[band]
    except KeyError:
        return ''


def band_from_hls_filename(filename: str) -> tuple[str, str]:
    # HLS.L30.T15TUG.2017167T165321.v2.0.B06.tif
    parts = filename.split('.')

    sensor_key, band = parts[1], parts[-2]

    return sensor_key, band


def band_from_rtc_filename(filename: str) -> tuple[str, str]:
    # OPERA_L2_RTC-S1_T063-133415-IW2_20170620T001327Z_20250925T045340Z_S1A_30_v1.0_VV.tif
    return 'RTC', filename.split('_')[-1].split('.')[0]

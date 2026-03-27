from datetime import datetime
from dataclasses import dataclass
from typing import NamedTuple
from typing import Tuple

import shapely


class Band(NamedTuple):
    id: str
    name: str
    shortname: str


def bands_by_shortname(bands: tuple[Band, ...]) -> dict[str, Band]:
    return {band.shortname: band for band in bands}


@dataclass(frozen=True)
class Modality:
    id: str
    bands: Tuple[Band, ...]

    collection: str


@dataclass(frozen=True)
class Event:
    name: str
    date: datetime
    wgs84_geometry: shapely.polygon


OPERA_RTC_BANDS_TUPLE = (
    Band("VV", "VV", "VV"),
    Band("VH", "VH", "VH"),
    Band("mask", "Validitiy Mask", "mask"),
)

OPERA_RTC_BANDS = bands_by_shortname(OPERA_RTC_BANDS_TUPLE)

# https://hyp3-docs.asf.alaska.edu/guides/opera_rtc_product_guide/
OPERA_RTC = Modality(
    id="RTC",
    bands=OPERA_RTC_BANDS_TUPLE,
    collection="OPERA_L2_RTC-S1_V1"
)


HLS_S30_BANDS_TUPLE = (
    Band("B02", "Blue", "B"),
    Band("B03", "Green", "G"),
    Band("B04", "Red", "R"),
    Band("B8A", "NIR Narrow", "N"),
    Band("B11", "SWIR 1", "SW1"),
    Band("B12", "SWIR 2", "SW2"),
    Band("Fmask", "Cloud Mask", "Fmask"),
)

HLS_S30_BANDS = bands_by_shortname(HLS_S30_BANDS_TUPLE)

# https://www.earthdata.nasa.gov/data/projects/hls/spectral-bands
HLS_S30 = Modality(
    id="HLS_S30",
    bands=HLS_S30_BANDS_TUPLE,
    collection="HLSS30"
)

HLS_L30_BANDS_TUPLE = (
    Band("B02", "Blue", "B"),
    Band("B03", "Green", "G"),
    Band("B04", "Red", "R"),
    Band("B05", "NIR Narrow", "N"),
    Band("B06", "SWIR 1", "SW1"),
    Band("B07", "SWIR 2", "SW2"),
    Band("Fmask", "Cloud Mask", "Fmask"),
)

HLS_L30_BANDS = bands_by_shortname(HLS_L30_BANDS_TUPLE)

HLS_L30 = Modality(
    id="HLS_L30",
    bands=HLS_L30_BANDS_TUPLE,
    collection="HLSL30"
)

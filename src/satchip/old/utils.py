import datetime
import functools
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, ParamSpec, TypeVar, TypedDict

import xarray as xr
import zarr
from pyproj import CRS, Transformer
from requests.exceptions import ConnectionError


class RtcImageSet(TypedDict):
    VV: Path
    VH: Path


class Bounds(NamedTuple):
    minx: float
    miny: float
    maxx: float
    maxy: float


class DateRange(NamedTuple):
    start: datetime.datetime
    end: datetime.datetime


class ChipDataRequiredOpts(TypedDict):
    strategy: str
    dates: DateRange | list[datetime.datetime]


class ChipDataOpts(ChipDataRequiredOpts, total=False):
    max_cloud_pct: int
    local_hyp3_paths: dict[str, list[RtcImageSet]]


def dates_error_msg(dates: DateRange | list[datetime.datetime]) -> str:
    return f'between {dates.start} and {dates.end}' if isinstance(dates, DateRange) else f'for dates {dates}'


def get_overall_bounds(bounds: list) -> Bounds:
    minx = min([b[0] for b in bounds])
    miny = min([b[1] for b in bounds])
    maxx = max([b[2] for b in bounds])
    maxy = max([b[3] for b in bounds])
    return Bounds(minx, miny, maxx, maxy)


def get_epsg4326_point(x: float, y: float, in_epsg: int) -> tuple[float, float]:
    if in_epsg == 4326:
        return x, y
    in_crs = CRS.from_epsg(in_epsg)
    out_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(in_crs, out_crs, always_xy=True)
    newx, newy = transformer.transform(x, y)
    return round(newx, 5), round(newy, 5)


def get_epsg4326_bbox(
    bounds: tuple[float, float, float, float], in_epsg: int, buffer: float = 0.1
) -> tuple[float, float, float, float]:
    minx, miny = get_epsg4326_point(bounds[0], bounds[1], in_epsg)
    maxx, maxy = get_epsg4326_point(bounds[2], bounds[3], in_epsg)
    bbox = minx - buffer, miny - buffer, maxx + buffer, maxy + buffer
    return bbox


def save_chip(dataset: xr.Dataset, save_path: str | Path) -> None:
    """Save a zipped zarr archive"""
    store = zarr.storage.ZipStore(save_path, mode='w')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Duplicate name:', module='zipfile')
        dataset.to_zarr(store)  # type: ignore[call-overload]
    store.close()


def load_chip(label_path: str | Path) -> xr.Dataset:
    """Load a zipped zarr archive"""
    store = zarr.storage.ZipStore(label_path, read_only=True)
    dataset = xr.open_zarr(store)
    return dataset


P = ParamSpec('P')
R = TypeVar('R')


def retry_on_connection_error(
    max_retries: int = 3, backoff_factor: float = 1
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, OSError) as e:
                    if attempt == max_retries - 1:
                        print(f'Failed after {max_retries} attempts: {e}')
                        raise e

                    wait_time = backoff_factor * (2**attempt)
                    time.sleep(wait_time)

            raise RuntimeError('Unexpected exit from retry loop')

        return wrapper

    return decorator

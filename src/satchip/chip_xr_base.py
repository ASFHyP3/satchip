import datetime
from collections.abc import Sequence

import numpy as np
import xarray as xr

import satchip
from satchip.terra_mind_grid import TerraMindChip


def _check_spec(dataset: xr.Dataset) -> None:
    assert isinstance(dataset, xr.Dataset)
    dims = ['band', 'time', 'x', 'y']
    assert sorted(list(dataset.dims)) == dims  # type: ignore
    coords = ['band', 'sample', 'spatial_ref', 'time', 'x', 'y']
    assert sorted(list(dataset.coords)) == coords  # type: ignore
    assert dataset.sample.ndim == 0
    data_vars = ['bands', 'center_lat', 'center_lon', 'crs']
    assert sorted(list(dataset.data_vars)) == data_vars
    assert 'date_created' in list(dataset.attrs.keys())
    assert 'satchip_version' in list(dataset.attrs.keys())
    assert 'bounds' in list(dataset.attrs.keys())


def create_dataset_chip(
    chip_array: np.ndarray,
    tm_chip: TerraMindChip,
    dates: list[datetime.datetime] | datetime.datetime,
    bands: Sequence[str],
) -> xr.Dataset:
    x = tm_chip.minx + (np.arange(tm_chip.nrow) + 0.5) * tm_chip.xres
    y = tm_chip.maxy + (np.arange(tm_chip.ncol) + 0.5) * tm_chip.yres
    if isinstance(dates, datetime.datetime):
        assert chip_array.ndim == 3, 'For single timestep, chip_array must have 3 dimensions (band, y, x)'
        dates = [dates]
        chip_array = chip_array.reshape(*(1, len(bands), tm_chip.ncol, tm_chip.nrow))
    else:
        assert chip_array.ndim == 4, 'For multiple timesteps, chip_array must have 4 dimensions (time, band, y, x)'
    coords = {'time': np.array(dates), 'band': np.array(bands), 'y': y, 'x': x}
    now = datetime.datetime.now().isoformat()
    dataset = xr.Dataset(attrs={'date_created': now, 'satchip_version': satchip.__version__})
    dataset.attrs['bounds'] = tm_chip.bounds
    dataset = dataset.assign_coords(sample=tm_chip.name)
    dataset = dataset.rio.write_crs(f'EPSG:{tm_chip.epsg}')
    dataset['bands'] = xr.DataArray(chip_array, coords=coords, dims=['time', 'band', 'y', 'x'])
    dataset['center_lat'] = xr.DataArray(tm_chip.center[1])
    dataset['center_lon'] = xr.DataArray(tm_chip.center[0])
    dataset['crs'] = xr.DataArray(tm_chip.epsg)
    _check_spec(dataset)
    return dataset


def create_template_da(chip: TerraMindChip) -> xr.DataArray:
    """Create a template DataArray with the same dimensions and transform as a label chip."""
    x = chip.minx + (np.arange(chip.nrow) + 0.5) * chip.xres
    y = chip.maxy + (np.arange(chip.ncol) + 0.5) * chip.yres
    template = xr.DataArray(np.zeros((chip.ncol, chip.nrow)), dims=('y', 'x'), coords={'y': y, 'x': x})
    template.rio.write_crs(f'EPSG:{chip.epsg}', inplace=True)
    template.rio.write_transform(chip.rio_transform, inplace=True)
    return template

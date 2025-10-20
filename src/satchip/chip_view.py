import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from satchip.utils import load_chip


def normalize_image_array(input_array: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Function to normalize array values to a byte value between 0 and 255

    Args:
        input_array: The array to normalize.
        vmin: The minimum value to normalize to (mapped to 0).
        vmax: The maximum value to normalize to (mapped to 255).

    Returns:
        The normalized array.
    """
    input_array = input_array.astype(float)
    scaled_array = (input_array - vmin) / (vmax - vmin)
    scaled_array[np.isnan(input_array)] = 0
    normalized_array = np.round(np.clip(scaled_array, 0, 1) * 255).astype(np.uint8)
    return normalized_array


def get_image_array(da: xr.Dataset, time_index: int, image_type: str, band: str | None) -> np.ndarray:
    timestep = da['bands'].isel(time=time_index)
    if image_type == 'rtc':
        vv = normalize_image_array(np.sqrt(timestep.sel(band='VV').data), 0.14, 0.52)
        vh = normalize_image_array(np.sqrt(timestep.sel(band='VH').data), 0.05, 0.259)
        img = np.stack([vv, vh, vv], axis=-1)
    elif image_type == 'optical':
        red = normalize_image_array(timestep.sel(band='RED').data, 0, 3000)
        green = normalize_image_array(timestep.sel(band='GREEN').data, 0, 3000)
        blue = normalize_image_array(timestep.sel(band='BLUE').data, 0, 3000)
        img = np.stack([red, green, blue], axis=-1)
    elif image_type == 'label':
        img = timestep.sel(band='LABEL').data
    elif image_type == 'user':
        assert band is not None, 'Band must be specified for user-defined image type'
        img = timestep.sel(band=band).data
    else:
        raise ValueError(f'Unknown image type: {image_type}')
    return img


def view_chip(label_path: Path, band: str | None) -> None:
    chip = load_chip(label_path)
    band_names = list(chip['band'].values)
    if band is not None:
        if band not in band_names:
            raise ValueError(f'Band {band} not found in chip. Available bands: {", ".join(band_names)}')
        image_type = 'user'
    elif any(b in band_names for b in ['VV', 'VH']):
        image_type = 'rtc'
    elif all(b in band_names for b in ['RED', 'GREEN', 'BLUE']):
        image_type = 'optical'
    elif 'LABEL' in band_names:
        image_type = 'label'
    else:
        raise ValueError('Cannot determine image type. Please specify a band using --band.')

    times = chip.time.values
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    time_index = 0
    im = ax.imshow(get_image_array(chip, time_index, image_type, band))
    title = ax.set_title(f'Date: {str(times[time_index]).split("T")[0]}')
    if len(times) > 1:
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])  # type: ignore
        slider = Slider(
            ax=ax_slider,
            label='Date Index',
            valmin=0,
            valmax=len(times) - 1,
            valinit=time_index,
            valstep=1,
        )

        def update(val: int) -> None:
            idx = int(slider.val)
            im.set_data(get_image_array(chip, idx, image_type, band))
            title.set_text(f'Time: {str(times[idx]).split("T")[0]}')
            f.canvas.draw_idle()

        slider.on_changed(update)  # type: ignore

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Chip a label image')
    parser.add_argument('chippath', type=Path, help='Path to the label image')
    parser.add_argument('--band', default=None, type=str, help='Band to view')
    args = parser.parse_args()
    view_chip(args.chippath, args.band)


if __name__ == '__main__':
    main()

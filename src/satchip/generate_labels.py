from pathlib import Path

import rasterio
from rasterio import features

from satchip import models


def binary_mask_from_template(template_data_path: Path, event: models.Event, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(exist_ok=True, parents=True)

    mask_path = output_dir / f'{event.name}.MASK.tif'

    with rasterio.open(template_data_path) as ds:
        profile = ds.profile

        mask_raster = features.rasterize(
            shapes=[[event.wgs84_geometry, 1]],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        profile.update(dtype='uint8', count=1, nodata=255)

        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask_raster, 1)
            print('generated:', mask_path)

    return mask_path

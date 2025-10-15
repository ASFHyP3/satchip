from datetime import datetime
from pathlib import Path

from osgeo import gdal

from satchip.chip_data import create_chips
from satchip.chip_label import chip_labels


gdal.UseExceptions()


def create_dataset(outpath: Path, start: tuple[int, int]) -> None:
    x, y = start
    pixel_size = 10
    cols, rows = 512, 512
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(str(outpath), cols, rows, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform((x, pixel_size, 0, y, 0, -pixel_size))
    dataset.SetProjection('EPSG:32611')
    array = dataset.GetRasterBand(1).ReadAsArray()
    array[:, :] = 0
    array[128:384, 128:384] = 1
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    dataset = None


if __name__ == '__main__':
    data_dir = Path('data')
    train_dir = data_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir = data_dir / 'val'
    val_dir.mkdir(parents=True, exist_ok=True)
    image_dir = data_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training data
    train_tif = data_dir / 'train.tif'
    create_dataset(train_tif, (431795, 3943142))
    chip_labels(train_tif, datetime.fromisoformat('20240115'), train_dir)
    for platform in ['S2L2A', 'HLS', 'S1RTC']:
        create_chips(
            list((train_dir / 'LABEL').glob('*.zarr.zip')),
            platform,
            datetime.fromisoformat('20240101'),
            datetime.fromisoformat('20240215'),
            'BEST',
            20,
            train_dir,
            image_dir,
        )
    
    # Create validation data
    val_tif = data_dir / 'val.tif'
    create_dataset(val_tif, (431795, 3943142 - 512 * 10))
    chip_labels(val_tif, datetime.fromisoformat('20240115'), val_dir)
    for platform in ['S2L2A', 'HLS', 'S1RTC']:
        create_chips(
            list((train_dir / 'LABEL').glob('*.zarr.zip')),
            platform,
            datetime.fromisoformat('20240101'),
            datetime.fromisoformat('20240215'),
            'BEST',
            20,
            val_dir,
            image_dir,
        )

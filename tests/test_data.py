from pathlib import Path

from osgeo import gdal


gdal.UseExceptions()


def create_dataset(dir=Path.cwd()) -> None:
    x = 431795
    y = 3943142
    pixel_size = 10
    cols, rows = 512, 512
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(str(dir / 'test.tif'), cols, rows, 1, gdal.GDT_UInt16)
    dataset.SetGeoTransform((x, pixel_size, 0, y, 0, -pixel_size))
    dataset.SetProjection('EPSG:32611')
    array = dataset.GetRasterBand(1).ReadAsArray()
    array[:, :] = 0
    array[128:384, 128:384] = 1
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()
    dataset = None


if __name__ == '__main__':
    create_dataset()

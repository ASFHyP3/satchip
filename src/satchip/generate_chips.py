import shutil
import zipfile
from pathlib import Path

import cartopy.crs as ccrs
import earthaccess
import gdown
import geopandas as gpd
import hls
import matplotlib.pyplot as plt
import mosaic
import numpy as np
import opera_rtc
import pandas as pd
import rasterio
from modality import Modality
from rasterio import features
from rasterio.windows import Window
from shapely.geometry import box
from sklearn.model_selection import train_test_split


QUITE = True
CHIP_SIZE = 256
RNG_SEED = 42


MODALITY = 'RTC'
ALL_BANDS = ('VV', 'VH', 'mask')
STACK_BANDS = ('VV', 'VH')
CHIP_BANDS = ('BANDS', 'EVENT', 'MASK')

MODALITY = 'HLS'
ALL_BANDS = ('B', 'G', 'R', 'N', 'SW1', 'SW2', 'Fmask')
STACK_BANDS = ('B', 'G', 'R', 'N', 'SW1', 'SW2')
CHIP_BANDS = ('BANDS', 'EVENT', 'MASK', 'Fmask')
SHOULD_CLEANUP = False


def main(modalities: list[Modality]):
    hwds_path = Path('hwds')

    print('Making folders')
    data_paths = {
        'CHIPS_ALL': hwds_path / 'CHIPS_ALL',
        'CHIPS': hwds_path / 'CHIPS',
        'MERGED': hwds_path / 'MERGED',
        'PLOTS': hwds_path / 'PLOTS',
    }

    if SHOULD_CLEANUP:
        for item in data_paths['MERGED'].glob('*.tif'):
            if item.is_dir():
                continue

            item.unlink()

        for p in ('CHIPS', 'CHIPS_ALL'):
            shutil.rmtree(data_paths[p], ignore_errors=True)

    for p in data_paths.values():
        p.mkdir(parents=True, exist_ok=True)

    gdf = _load_event_database(hwds_path)
    gdf_utm = gdf.to_crs(32615)
    gdf['buffered_event'] = gdf_utm.buffer(3000).to_crs(4326)
    gdf['buffered_event_background'] = gdf_utm.buffer(10000).to_crs(4326)
    gdf = gdf.to_crs(4326)

    # keepers = [1442, 622, 1079, 628]
    keepers = [1442, 622]
    gdf = gdf[gdf['swathID'].isin(keepers)]

    earthaccess.login()

    tm_chips = []

    for i, (swathID, swath) in enumerate(gdf.iterrows(), start=1):
        swathID = f'{int(swath["swathID"]):04d}'
        print(f'Processing  Swath {swathID} ({i} / {len(gdf)})')

        merged = mosaic.data_over_swath(swath, modalities, output_path=data_paths['MERGED'])

        template_path = merged[modalities[0].id]['BANDS']
        event_tif, mask_tif = _generate_masks(template_path, swathID, swath)

        merged_data = {
            **merged,
            'EVENT': event_tif,
            'MASK': mask_tif,
        }

        if not all(is_valid_data(merged_data, modality) for modality in modalities):
            print('Skipping: not enough valid data')
            continue

        for modality in modalities:
            print(f'Chipping {modality.id}!')
            chips = _chip_data(merged_data, data_paths['CHIPS_ALL'], modality)

            good_chips = filter_chips(chips, modality)
            print(f'Found {len(good_chips)} good chips')

            for chip in good_chips:
                for band, chip_path in chip.items():
                    if band not in ('MASK', 'BANDS'):
                        continue

                    dest = data_paths['CHIPS'] / chip_path.name
                    shutil.copy(chip_path, dest)

            tm_chips += good_chips

    for _, swath in gdf.iterrows():
        swath_id = _make_swath_id(swath['swathID'])

        for modality in modalities:
            merged_file = list(data_paths['MERGED'].glob(f'{swath_id}.{modality.id}.*.BANDS.tif'))

            if len(merged_file) == 0:
                print(f'no chips for {swath_id}')
                continue

            all_chips = list(data_paths['CHIPS_ALL'].glob(f'*.{swath_id}.{modality.id}.*.tif'))
            good_chips = list(data_paths['CHIPS'].glob(f'*.{swath_id}.{modality.id}.*.tif'))

            print(f'plotting {swath_id}')
            _plot_chips(merged_file[0], all_chips, good_chips, swath, modality, save_to=data_paths['PLOTS'])

    for modality in modalities:
        print(f'Calulating stats for modality: {modality.id}')
        band_chips = list(data_paths['CHIPS'].glob(f'*.{modality.id}.*.BANDS.tif'))

        means, stds = calculate_stats(chips=band_chips, n_bands=len(modality.stack_bands))

        means_str = ', '.join(f'{x:.4f}' for x in means)
        stds_str = ', '.join(f'{x:.4f}' for x in stds)

        stats_str = f'Means {modality.stack_bands}: {means_str}\nStds {modality.stack_bands}: {stds_str}\n'
        (hwds_path / '{modality.id}-statistics.txt').write_text(stats_str)
        print(stats_str)


def _generate_masks(template_data_path: Path, swathID: str, swath: pd.Series) -> tuple[Path, Path]:
    event_path = template_data_path.parent / f'{swathID}.EVENT.tif'
    mask_path = template_data_path.parent / f'{swathID}.MASK.tif'

    with rasterio.open(template_data_path) as ds:
        profile = ds.profile

        mask_raster = features.rasterize(
            shapes=[[swath['geometry'], 1]],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(mask_path, 'w', **profile) as dst:
            dst.write(mask_raster, 1)
            print('generated:', mask_path)

        event_mask = features.rasterize(
            shapes=[
                [swath['buffered_event_background'], 3],
                [swath['buffered_event'], 2],
                [swath['geometry'], 1],
            ],
            fill=0,
            out_shape=ds.shape,
            transform=ds.transform,
        )

        with rasterio.open(event_path, 'w', **profile) as dst:
            dst.write(event_mask, 1)
            print('generated:', event_path)

    return event_path, mask_path


def _load_event_database(data_dir: Path):
    # use 60-swath version
    hwds_google_drive_id = '1h_JIEcrrUF3OSTrmwAKNPa0eUEhPA2Xx'
    drive_url = f'https://drive.google.com/uc?id={hwds_google_drive_id}'

    shp_dir = data_dir / 'SHP'
    shp_dir.mkdir(parents=True, exist_ok=True)

    filename = 'hwds_v3_20250205_subset_60.zip'

    zip_path = shp_dir / filename

    if not zip_path.exists():
        gdown.download(drive_url, str(zip_path), quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=shp_dir)

    shp_path = shp_dir / 'hwds_v3_20250205_subset_60.shp'
    gdf = gpd.read_file(shp_path)

    gdf['swathDate'] = pd.to_datetime(gdf['swathDate'], format='%Y-%m-%d')
    gdf['ls5hlsDate'] = pd.to_datetime(gdf['ls5hlsDate'], format='%Y-%m-%d')
    gdf['s1Date'] = pd.to_datetime(gdf['s1Date'], format='%Y-%m-%d')

    return gdf


def create_split_files(band_chips: list[Path], splits_path: Path) -> None:
    chip_ids = [p.name.removesuffix('BANDS.tif') for p in band_chips]

    the_rest, test = train_test_split(chip_ids, test_size=0.15, random_state=RNG_SEED)
    train, val = train_test_split(the_rest, test_size=0.15, random_state=RNG_SEED)

    splits = {'train': train, 'val': val, 'test': test}

    for split, chip_ids in splits.items():
        split_path = splits_path / f'{split}.txt'
        split_path.write_text('\n'.join(chip_ids))


def calculate_stats(chips: list[Path], n_bands: int = 2) -> tuple:
    mean = np.zeros(n_bands, dtype=np.float64)
    M2 = np.zeros(n_bands, dtype=np.float64)
    count = np.zeros(n_bands, dtype=np.float64)

    for chip in chips:
        with rasterio.open(chip) as src:
            band_data = src.read()
            count, mean, M2 = 0, 0, 0

            _, H, W = band_data.shape

            batch_count = H * W
            batch_mean = band_data.mean(axis=(1, 2))
            batch_var = band_data.var(axis=(1, 2))

            delta = batch_mean - mean
            total_count = count + batch_count

            mean = mean + delta * (batch_count / total_count)
            M2 = M2 + batch_var * batch_count + (delta**2) * count * batch_count / total_count
            count = total_count

    variance = M2 / count
    std = np.sqrt(variance)

    return mean, std


def _plot_chips(
    merged_band_file,
    all_chips,
    good_chips,
    swath,
    modality,
    save_to: Path | None = None,
    quite=QUITE,
):
    crs_pc = ccrs.PlateCarree()

    with rasterio.open(merged_band_file) as ds:
        bounds = ds.bounds
        full_extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        band_data = ds.read()

    img = get_img(band_data, modality)

    # plot BANDS and geom
    fig, ax = plt.subplots(
        1,
        1,
        subplot_kw={'projection': crs_pc},
        figsize=(12, 12),
        layout='constrained',
    )

    swath_geom = swath['geometry']

    ax.imshow(img, extent=full_extent, origin='upper', transform=crs_pc)
    ax.add_geometries([swath_geom], edgecolor='red', linewidth=2, facecolor='none', crs=crs_pc)

    def show_chips(chips, color, linewidth, z):
        for chip in chips:
            with rasterio.open(chip) as ds:
                chip_bounds = ds.bounds
                chip_geom = box(
                    chip_bounds.left,
                    chip_bounds.bottom,
                    chip_bounds.right,
                    chip_bounds.top,
                )

            ax.add_geometries(
                [chip_geom],
                edgecolor=color,
                linewidth=linewidth,
                alpha=1,
                zorder=z,
                facecolor='none',
                crs=crs_pc,
            )

    show_chips(all_chips, 'yellow', 1, z=1)
    show_chips(good_chips, 'blue', 3, z=2)

    ax.set_extent(full_extent, crs=crs_pc)

    if save_to:
        plt.savefig(
            save_to / f'{merged_band_file.name.removesuffix("BANDS.tif")}.png',
            dpi=300,
            bbox_inches='tight',
        )

    if not quite:
        plt.show()

    plt.close(fig)


def _make_swath_id(swathID):
    return f'{int(swathID):04d}'


def _chip_data(merged, output_path: Path, modality: Modality, chip_size=CHIP_SIZE):
    chips = {}
    grid = []

    with rasterio.open(merged[modality.id]['BANDS']) as ref:
        n_cols = ref.width // chip_size
        n_rows = ref.height // chip_size

        for row in range(n_rows):
            for col in range(n_cols):
                window = Window(col * chip_size, row * chip_size, chip_size, chip_size)
                bounds = ref.window_bounds(window)

                tile_id = f'{row:03d}.{col:03d}'
                chips[tile_id] = {}
                grid.append((tile_id, bounds))

    for chip_layer in modality.chip_bands:
        if chip_layer in merged[modality.id]:
            layer_path = merged[modality.id][chip_layer]
        else:
            layer_path = merged[chip_layer]

        with rasterio.open(layer_path) as src:
            for tile_id, bounds in grid:
                window = src.window(*bounds)
                window = Window(
                    round(window.col_off),
                    round(window.row_off),
                    round(window.width),
                    round(window.height),
                )

                data = src.read(window=window)

                if chip_layer == 'BANDS':
                    data = data_transform(data, modality)

                chip_meta = src.meta.copy()
                chip_meta.update(
                    {
                        'width': window.width,
                        'height': window.height,
                        'transform': src.window_transform(window),
                    }
                )

                chip_name = f'{tile_id}.{layer_path.name}'
                chip_path = output_path / chip_name

                with rasterio.open(chip_path, 'w', **chip_meta) as dst:
                    dst.write(data)

                chips[tile_id][chip_layer] = chip_path

    return chips


def is_valid_data(merged, modality):
    merged_data = merged[modality.id]

    if modality.id == 'HLS':
        is_valid = hls.is_valid_hls(merged_data['Fmask'], merged['EVENT'])
    elif modality.id == 'RTC':
        is_valid = opera_rtc.is_valid_rtc(merged_data['mask'], merged['EVENT'])

    return is_valid


def filter_chips(chips, modality):
    if modality.id == 'HLS':
        filtered_chips = hls.filter_hls_chips(chips)
    elif modality.id == 'RTC':
        filtered_chips = opera_rtc.filter_rtc_chips(chips)

    return filtered_chips


def get_img(band_data, modality):
    if modality.id == 'HLS':
        img = hls.get_hls_img(band_data)
    elif modality.id == 'RTC':
        img = opera_rtc.get_rtc_img(band_data)

    return img


def data_transform(data, modality):
    if modality.id == 'RTC':
        data = 10 * np.log10(np.clip(data, 1e-10, None))

    return data


if __name__ == '__main__':
    opera_rtc_mod = Modality(
        id='RTC', all_bands=('VV', 'VH', 'mask'), stack_bands=('VV', 'VH'), chip_bands=('BANDS', 'EVENT', 'MASK')
    )

    hls_mod = Modality(
        id='HLS',
        all_bands=('B', 'G', 'R', 'N', 'SW1', 'SW2', 'Fmask'),
        stack_bands=('B', 'G', 'R', 'N', 'SW1', 'SW2'),
        chip_bands=('BANDS', 'EVENT', 'MASK', 'Fmask'),
    )

    modalities = [hls_mod, opera_rtc_mod]

    main(modalities)

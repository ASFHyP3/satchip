from satchip import merge_modality, models
import rasterio


def test_make_merge_name(s2_event):
    merged_name = merge_modality._make_merge_name(s2_event.name, s2_event.date, 'BAND', 'MOD')

    assert 'BAND' in merged_name
    assert 'MOD' in merged_name
    assert s2_event.name in merged_name
    assert merged_name.endswith('.tif')


def test_merge_s2_modality_empty(s2_local_files, s2_event, tmp_path):
    empty_result = merge_modality.merge_modality([], models.HLS_S30, s2_event, tmp_path)
    assert len(empty_result) == 0


def test_merge_s2_modality(s2_local_files, s2_event, tmp_path):
    merged_files = merge_modality.merge_modality(s2_local_files, models.HLS_S30, s2_event, tmp_path / 'merged')

    assert len(merged_files) == len(models.HLS_S30['bands'])
    assert all('merged' in str(result) for result in merged_files)

    shapes = set()
    for merged_file in merged_files:
        with rasterio.open(merged_file) as ds:
            shapes.add(ds.shape)

    assert len(shapes) == 1


def test_stack_bands(s2_local_files, s2_event, tmp_path):
    bands = tuple(b for b in models.HLS_S30['bands'] if b.id != 'Fmask')

    merged_bands = merge_modality.merge_modality(s2_local_files, models.HLS_S30, s2_event, tmp_path / 'merged', bands)
    stacked_data = merge_modality.stack_bands(merged_bands, stacked_filename=tmp_path / 'stacked.tif')

    assert 'stacked.tif' in stacked_data.name

    with rasterio.open(stacked_data) as ds:
        num_bands = ds.count

    assert num_bands == len(bands)

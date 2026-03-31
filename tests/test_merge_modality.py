from satchip import merge_modality, models


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
    result = merge_modality.merge_modality(s2_local_files, models.HLS_S30, s2_event, tmp_path)

    assert result

    result = merge_modality.merge_modality(s2_local_files, models.HLS_S30, s2_event, tmp_path / 'merged')

    assert result

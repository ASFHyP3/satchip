from satchip import merge_modality, models


def test_merge_s2_modality(s2_local_files, s2_event):
    empty_result = merge_modality.merge_modality([], s2_event, models.HLS_S30_BANDS['R'], models.HLS_S30)
    assert len(empty_result) == 0

    result = merge_modality.merge_modality(s2_local_files, s2_event, models.HLS_S30_BANDS['R'], models.HLS_S30)

    breakpoint()

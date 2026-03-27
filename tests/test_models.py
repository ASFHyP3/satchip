from satchip import models


def test_hls_model_len():
    assert len(models.HLS_S30_BANDS_TUPLE) == 7
    assert len(models.HLS_L30_BANDS_TUPLE) == 7


def test_hls_model_names():
    assert models.HLS_S30_BANDS['N'].id != models.HLS_L30_BANDS['N'].id
    assert models.HLS_S30_BANDS['SW1'].id != models.HLS_L30_BANDS['SW1'].id
    assert models.HLS_S30_BANDS['SW2'].id != models.HLS_L30_BANDS['SW2'].id

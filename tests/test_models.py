import pytest

from satchip import models


def test_hls_model_len():
    assert len(models.HLS_S30_BANDS) == 7
    assert len(models.HLS_L30_BANDS) == 7


def test_hls_model_names():
    assert models.HLS_S30_BANDS['N'].id != models.HLS_L30_BANDS['N'].id
    assert models.HLS_S30_BANDS['SW1'].id != models.HLS_L30_BANDS['SW1'].id
    assert models.HLS_S30_BANDS['SW2'].id != models.HLS_L30_BANDS['SW2'].id


@pytest.mark.parametrize(
    'filename, expected_band',
    [
        ('OPERA_L2_RTC-S1_T085-181260-IW1_20190822T125312Z_20250913T222203Z_S1A_30_v1.0_VH.tif', 'VH'),
        ('OPERA_L2_RTC-S1_T165-352512-IW3_20200723T000516Z_20250908T213809Z_S1B_30_v1.0_VH.tif', 'VH'),
        ('OPERA_L2_RTC-S1_T085-181260-IW1_20190822T125312Z_20250913T222203Z_S1A_30_v1.0_VV.tif', 'VV'),
        ('OPERA_L2_RTC-S1_T165-352512-IW3_20200723T000516Z_20250908T213809Z_S1B_30_v1.0_VV.tif', 'VV'),
        ('OPERA_L2_RTC-S1_T085-181260-IW1_20190822T125312Z_20250913T222203Z_S1A_30_v1.0_mask.tif', 'mask'),
        ('OPERA_L2_RTC-S1_T165-352512-IW3_20200723T000516Z_20250908T213809Z_S1B_30_v1.0_mask.tif', 'mask'),
    ],
)
def test_band_id_from_filename_rtc(filename, expected_band):
    assert models.band_id_from_filename(filename, 'OPERA_RTC').id == expected_band


@pytest.mark.parametrize(
    'filename, expected_band',
    [
        ('HLS.S30.T13TFL.2019187T174919.v2.0.B12.tif', 'B12'),
        ('HLS.S30.T13TGM.2018184T173901.v2.0.B11.tif', 'B11'),
        ('HLS.S30.T14TLQ.2019219T173911.v2.0.B8A.tif', 'B8A'),
        ('HLS.S30.T15TXG.2017167T170311.v2.0.B04.tif', 'B04'),
        ('HLS.S30.T13TFL.2019187T174919.v2.0.Fmask.tif', 'Fmask'),
        ('HLS.S30.T13TGM.2018184T173901.v2.0.B12.tif', 'B12'),
        ('HLS.S30.T14TLQ.2019219T173911.v2.0.B11.tif', 'B11'),
        ('HLS.S30.T15TXG.2017167T170311.v2.0.B8A.tif', 'B8A'),
        ('HLS.S30.T13TFL.2020157T174911.v2.0.B02.tif', 'B02'),
        ('HLS.S30.T13TGM.2018184T173901.v2.0.Fmask.tif', 'Fmask'),
        ('HLS.S30.T13TFL.2020157T174911.v2.0.B03.tif', 'B03'),
        ('HLS.S30.T14SKH.2019211T172909.v2.0.B02.tif', 'B02'),
        ('HLS.S30.T14TLQ.2019219T173911.v2.0.Fmask.tif', 'Fmask'),
        ('HLS.S30.T13TFL.2020157T174911.v2.0.B04.tif', 'B04'),
        ('HLS.S30.T14SKH.2019211T172909.v2.0.B03.tif', 'B03'),
        ('HLS.S30.T14TLQ.2020156T172859.v2.0.B02.tif', 'B02'),
        ('HLS.S30.T15TXG.2017167T170311.v2.0.Fmask.tif', 'Fmask'),
    ],
)
def test_band_id_from_filename_hls_s30(filename, expected_band):
    assert models.band_id_from_filename(filename, 'HLS_S30').id == expected_band

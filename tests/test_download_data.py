import pytest

from satchip import models
from satchip import download_data


def test_models():
    assert len(models.HLS_S30_BANDS) == 7
    assert len(models.HLS_L30_BANDS) == 7


@pytest.mark.download
def test_download_data(s2_event, tmp_path):
    local_files = download_data.download_data(s2_event, models.HLS_S30, tmp_path)
    assert len(local_files) > 1

    local_files = download_data.download_data(s2_event, models.HLS_L30, tmp_path)
    assert len(local_files) == 0

import pytest

from satchip import download_data, models


@pytest.mark.download
def test_download_data(s2_event, tmp_path):
    local_files = download_data.download_data(s2_event, models.HLS_S30, tmp_path)
    assert len(local_files) > 1

    local_files = download_data.download_data(s2_event, models.HLS_L30, tmp_path)
    assert len(local_files) == 0

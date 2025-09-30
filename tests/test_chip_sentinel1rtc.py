import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from shapely.geometry import box, mapping

from satchip import chip_sentinel1rtc


def test_bounds_check():
    chip_sentinel1rtc._check_bounds_size([0, 0, 1, 1])
    chip_sentinel1rtc._check_bounds_size([0, 0, 2.9, 1])
    chip_sentinel1rtc._check_bounds_size([-107.79192, 45.74287, -105.01543, 46.48598])

    with pytest.raises(AssertionError):
        chip_sentinel1rtc._check_bounds_size([0, 0, 3, 1])


def test_get_granules():
    bounds = [-107.79192, 45.74287, -105.01543, 46.48598]
    date_start = datetime.datetime(2020, 7, 7)
    date_end = date_start + datetime.timedelta(days=14)

    mock_search_result = MagicMock()
    mock_search_result = ['granule1', 'granule2']

    with patch('satchip.chip_sentinel1rtc.asf.geo_search', return_value=mock_search_result) as mock_geo_search:
        results = chip_sentinel1rtc._get_granules(bounds, date_start, date_end)

        mock_geo_search.assert_called_once()

        assert results == mock_search_result

        args, kwargs = mock_geo_search.call_args
        assert (
            kwargs['intersectsWith']
            == 'POLYGON ((-105.01543 45.74287, -105.01543 46.48598, -107.79192 46.48598, -107.79192 45.74287, -105.01543 45.74287))'
        )
        assert kwargs['start'] == date_start
        assert kwargs['end'] == date_end + datetime.timedelta(days=1)


def test_get_slcs_for_each_chip_custom_intersect():
    granule1 = MagicMock()
    granule1.geometry = mapping(box(0, 0, 2, 2))
    granule1.properties = {'startTime': '2025-01-01T00:00:00Z'}

    granule2 = MagicMock()
    granule2.geometry = mapping(box(3, 3, 5, 5))
    granule2.properties = {'startTime': '2025-01-02T00:00:00Z'}

    granule3 = MagicMock()
    granule3.geometry = mapping(box(10, 10, 15, 15))
    granule3.properties = {'startTime': '2025-01-03T00:00:00Z'}

    chip1 = MagicMock()
    chip1.name = 'chip1'
    chip1.bounds = [0, 0, 1, 1]

    chip2 = MagicMock()
    chip2.name = 'chip2'
    chip2.bounds = [1, 1, 2, 2]

    chip3 = MagicMock()
    chip3.name = 'chip3'
    chip3.bounds = [3, 3, 4, 4]

    chips = [chip1, chip2, chip3]
    granules = [granule1, granule2, granule3]

    result = chip_sentinel1rtc._get_slcs_for_each_chip(chips, granules, strategy='BEST')

    assert result['chip1'] == [granule1]
    assert result['chip2'] == [granule1]
    assert result['chip3'] == [granule2]


def test_get_slcs_for_each_chip_with_different_strategies():
    granule1 = MagicMock()
    granule1.geometry = mapping(box(0, 0, 1, 1))
    granule1.properties = {'startTime': '2025-01-01T00:00:00Z'}

    granule2 = MagicMock()
    granule2.geometry = mapping(box(0, 0, 5, 5))
    granule2.properties = {'startTime': '2025-01-02T00:00:00Z'}

    granule3 = MagicMock()
    granule3.geometry = mapping(box(0, 0, 15, 15))
    granule3.properties = {'startTime': '2025-01-03T00:00:00Z'}

    chip1 = MagicMock()
    chip1.name = 'chip1'
    chip1.bounds = [0, 0, 5, 10]

    chips = [chip1]
    granules = [granule1, granule2, granule3]

    result = chip_sentinel1rtc._get_slcs_for_each_chip(chips, granules, strategy='BEST', intersection_pct=49)
    assert result['chip1'] == [granule3]

    result = chip_sentinel1rtc._get_slcs_for_each_chip(chips, granules, strategy='ALL', intersection_pct=49)
    assert result['chip1'] == [granule3, granule2]


def test_get_slcs_for_each_chip_no_matches():
    chip = MagicMock()
    chip.name = 'chip1'
    chip.bounds = [0, 0, 1, 1]

    with pytest.raises(ValueError, match='No products found for chip chip1'):
        chip_sentinel1rtc._get_slcs_for_each_chip([chip], [], strategy='BEST')


class MockS1Product:
    def __init__(self, scene_name: str):
        self.properties = {'sceneName': scene_name}


def test_get_rtcs_for():
    slcs_for_chips = {
        'chip_001': [MockS1Product('SLC_1'), MockS1Product('SLC_2')],
        'chip_002': [MockS1Product('SLC_3'), MockS1Product('SLC_4')],
    }
    scratch_dir = Path('/tmp')

    mock_jobs = []
    for slc_name in ['SLC_1', 'SLC_2', 'SLC_3', 'SLC_4']:
        job = MagicMock()
        job.job_parameters = {'granules': [slc_name]}
        mock_jobs.append(job)

    with (
        patch('satchip.chip_sentinel1rtc._process_rtcs', return_value=mock_jobs) as mock_process_rtcs,
        patch('satchip.chip_sentinel1rtc._download_hyp3_rtc') as mock_download,
    ):

        def mock_download_fn(job, scratch):
            return Path(f'/tmp/{job.job_parameters["granules"][0]}_rtc.tif')

        mock_download.side_effect = mock_download_fn

        result = chip_sentinel1rtc._get_rtcs_for(slcs_for_chips, scratch_dir)

        expected = {
            'chip_001': [Path('/tmp/SLC_1_rtc.tif'), Path('/tmp/SLC_2_rtc.tif')],
            'chip_002': [Path('/tmp/SLC_3_rtc.tif'), Path('/tmp/SLC_4_rtc.tif')],
        }

        assert result == expected
        mock_process_rtcs.assert_called_once_with({'SLC_1', 'SLC_2', 'SLC_3', 'SLC_4'})
        assert mock_download.call_count == 4

import datetime
from unittest.mock import MagicMock, patch

import pytest

from satchip import chip_sentinel1rtc


def test_bounds_check():
    chip_sentinel1rtc.check_bounds_size([0, 0, 1, 1])
    chip_sentinel1rtc.check_bounds_size([0, 0, 2.9, 1])
    chip_sentinel1rtc.check_bounds_size([-107.79192, 45.74287, -105.01543, 46.48598])

    with pytest.raises(AssertionError):
        chip_sentinel1rtc.check_bounds_size([0, 0, 3, 1])


def test_get_granules():
    bounds = [-107.79192, 45.74287, -105.01543, 46.48598]
    date_start = datetime.datetime(2020, 7, 7)
    date_end = date_start + datetime.timedelta(days=14)

    mock_search_result = MagicMock()
    mock_search_result.items = ['granule1', 'granule2']

    with patch('satchip.chip_sentinel1rtc.asf.geo_search', return_value=mock_search_result) as mock_geo_search:
        results = chip_sentinel1rtc.get_granules(bounds, date_start, date_end)

        mock_geo_search.assert_called_once()

        assert results == mock_search_result

        args, kwargs = mock_geo_search.call_args
        assert (
            kwargs['intersectsWith']
            == 'POLYGON ((-105.01543 45.74287, -105.01543 46.48598, -107.79192 46.48598, -107.79192 45.74287, -105.01543 45.74287))'
        )
        assert kwargs['start'] == date_start
        assert kwargs['end'] == date_end + datetime.timedelta(days=1)

import pytest
import datetime

from satchip import chip_data


def test_bounds_check():
    chip_data.check_bounds_size([0, 0, 1, 1])
    chip_data.check_bounds_size([0, 0, 2.9, 1])
    chip_data.check_bounds_size([-107.79192, 45.74287, -105.01543, 46.48598])

    with pytest.raises(AssertionError):
        chip_data.check_bounds_size([0, 0, 3, 1])


def test_get_granules():
    start_date = datetime.datetime(2020, 7, 7)
    end_date = start_date + datetime.timedelta(days=14)

    assert chip_data.get_granules([-107.79192, 45.74287, -105.01543, 46.48598], start_date, end_date)


def test_pair_slcs_to_chips():
    pass

from collections import namedtuple

from satchip.chip_sentinel2 import get_latest_image_versions


ItemStub = namedtuple('ItemStub', ['id', 'properties'])


def test_get_latest_image_versions():
    items = [
        ItemStub(id='S2B_13TEG_20190623_0_L2A', properties={'s2:sequence': 0}),
        ItemStub(id='S2B_13TEG_20190623_1_L2A', properties={'s2:sequence': 1}),
        ItemStub(id='S2A_13TEG_20190621_0_L2A', properties={'s2:sequence': 0}),
        ItemStub(id='S2A_13TEG_20190618_0_L2A', properties={'s2:sequence': 0}),
        ItemStub(id='S2A_13TEG_20190618_1_L2A', properties={'s2:sequence': 1}),
        ItemStub(id='S2A_13TEG_20190618_3_L2A', properties={'s2:sequence': 2}),
    ]

    latest_items = get_latest_image_versions(items)  # type: ignore

    assert len(latest_items) == 3
    assert any(item.id == 'S2B_13TEG_20190623_1_L2A' for item in latest_items)
    assert any(item.id == 'S2A_13TEG_20190621_0_L2A' for item in latest_items)
    assert any(item.id == 'S2A_13TEG_20190618_3_L2A' for item in latest_items)

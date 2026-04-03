import rasterio

from satchip import chip_data


def test_make_grid(warped_event_files):
    label = [f for f in warped_event_files if 'MASK' in f.name].pop()

    grid = chip_data.make_grid_from_reference(label)
    assert len(grid) == 27

    grid = chip_data.make_grid_from_reference(label, chip_size=512)
    assert len(grid) == 4

    grids = (tuple(chip_data.make_grid_from_reference(f)) for f in warped_event_files)
    assert len(set(grids)) == 1


def test_chip_data(warped_event_files, tmp_path):
    label = [f for f in warped_event_files if 'MASK' in f.name].pop()
    grid = chip_data.make_grid_from_reference(label)

    for file in warped_event_files:
        chips = chip_data.chip_data(grid, file, tmp_path / 'chips')

        chip = chips[0]
        assert file.name in chip.path.name
        assert chip.id in chip.path.name

        assert len(chips) == len(grid)

        shapes = set()
        for chip in chips:
            with rasterio.open(chip.path) as ds:
                shapes.add(ds.shape)

        assert shapes == {(256, 256)}

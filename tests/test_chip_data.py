import rasterio

from satchip import chip_data, models


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


def test_make_chip_stacks(warped_event_files, tmp_path):
    fmask, label, data = warped_event_files
    grid = chip_data.make_grid_from_reference(fmask)

    fmask_chips = chip_data.chip_data(grid, fmask, tmp_path / 'chips')
    label_chips = chip_data.chip_data(grid, label, tmp_path / 'chips')
    data_chips = chip_data.chip_data(grid, data, tmp_path / 'chips')

    stacks = chip_data.make_chip_stacks(data_chips, fmask_chips, label_chips, models.HLS_S30)

    assert len(stacks) == len(fmask_chips)

    for stack in stacks:
        assert stack.id in stack.validation_mask.name
        assert stack.id in stack.data.name
        assert stack.id in stack.label.name
        assert stack.modality == models.HLS_S30


def test_filter_chips(warped_event_files, tmp_path):
    fmask, label, data = warped_event_files
    grid = chip_data.make_grid_from_reference(fmask)

    fmask_chips = chip_data.chip_data(grid, fmask, tmp_path / 'chips')
    label_chips = chip_data.chip_data(grid, label, tmp_path / 'chips')
    data_chips = chip_data.chip_data(grid, data, tmp_path / 'chips')

    stacks = chip_data.make_chip_stacks(data_chips, fmask_chips, label_chips, models.HLS_S30)

    filtered = chip_data.filter_chips(stacks)

    assert len(filtered) > 0
    assert len(filtered) < len(stacks)

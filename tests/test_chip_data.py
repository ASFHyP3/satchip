from satchip import chip_data


def test_make_grid(warped_event_files):
    label = [f for f in warped_event_files if 'MASK' in f.name].pop()

    grid = chip_data.make_grid_from_reference(label)
    assert len(grid) == 27

    grid = chip_data.make_grid_from_reference(label, chip_size=512)
    assert len(grid) == 4

    grids = (tuple(chip_data.make_grid_from_reference(f)) for f in warped_event_files)
    assert len(set(grids)) == 1


def test_chip_data(warped_event_files):
    label = [f for f in warped_event_files if 'MASK' in f.name].pop()
    grid = chip_data.make_grid_from_reference(label)

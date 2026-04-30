from satchip import view, merge_modality, chip_data, models


def test_view_merged(s2_local_files, s2_event, tmp_path):
    bands = tuple(b for b in models.HLS_S30['bands'] if b.id != 'Fmask')

    merged_bands = merge_modality.merge_modality(s2_local_files, models.HLS_S30, s2_event, tmp_path / 'merged', bands)
    stacked_data = merge_modality.stack_bands(merged_bands, stacked_filename=tmp_path / 'stacked.tif')
    reprojected = merge_modality.reproject_files([stacked_data], tmp_path / 'reproj')[0]

    view.view_merged(
        reprojected, s2_event, models.HLS_S30, rgb_bands=[2, 1, 0], save_to_file=tmp_path / 'output.png', quite=False
    )


def test_view_chip(warped_event_files, tmp_path):
    fmask, label, data = warped_event_files
    grid = chip_data.make_grid_from_reference(fmask)

    fmask_chips = chip_data.chip_data(grid, fmask, tmp_path / 'chips')
    label_chips = chip_data.chip_data(grid, label, tmp_path / 'chips')
    data_chips = chip_data.chip_data(grid, data, tmp_path / 'chips')

    stacks = chip_data.make_chip_stacks(data_chips, fmask_chips, label_chips, models.HLS_S30)
    filtered = chip_data.filter_chips(stacks)

    view.view_chip(filtered[0], models.HLS_S30, [2, 1, 0])


def test_view_chips(warped_event_files, tmp_path):
    fmask, label, data = warped_event_files
    grid = chip_data.make_grid_from_reference(fmask)

    fmask_chips = chip_data.chip_data(grid, fmask, tmp_path / 'chips')
    label_chips = chip_data.chip_data(grid, label, tmp_path / 'chips')
    data_chips = chip_data.chip_data(grid, data, tmp_path / 'chips')

    stacks = chip_data.make_chip_stacks(data_chips, fmask_chips, label_chips, models.HLS_S30)
    filtered = chip_data.filter_chips(stacks)

    view.view_chips(filtered, models.HLS_S30, [2, 1, 0])

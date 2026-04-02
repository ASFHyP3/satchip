import rasterio

from satchip import generate_labels


def test_generate_labales(s2_local_files, s2_event, tmp_path):
    template_file = s2_local_files[0]

    output = generate_labels.binary_mask_from_template(template_file, s2_event, tmp_path / 'labels')

    assert output.name.endswith('MASK.tif')

    with rasterio.open(output) as mask_ds:
        with rasterio.open(template_file) as template_ds:
            assert mask_ds.shape == template_ds.shape

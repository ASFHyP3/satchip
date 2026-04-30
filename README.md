# SatChip

A package for satellite image AI data prep.

## Usage

```python
import satchip

data_paths = {
    MODALITY: hwds_path / MODALITY,
    "RAW": modality_path / "RAW",
    "WGS84": modality_path / "WGS84",
    "MERGE": modality_path / "MERGE",
    "CHIPS": modality_path / "CHIPS",
    "CHIPS_TM": modality_path / "CHIPS_TM",
    "PLOTS": modality_path / "PLOTS",
    "SPLITS": modality_path / "SPLITS",
}

modalities = ['HLS']

data = satchip.find_data(area, modality)
reprojected_data = satchip.repoject(raw_data, projection='WGS84', modality=)  # stack bands and mosaic
mosaics = satchip.mosaic(data, stack_bands=True)  # stack bands and mosaic
masks = satchip.generate_masks(mosaics)

chips = satchip.chip_data(mosaics, masks)
chips = satchip.filter_chips(chips)
```

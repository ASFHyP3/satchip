from pathlib import Path

from satchip import models


def merge_modality(band_tifs: list[Path], event: models.Event, band: models.Band, modality: models.Modality):
    if len(band_tifs) == 0:
        print(f"Warning: no data for {event.name}")
        return []

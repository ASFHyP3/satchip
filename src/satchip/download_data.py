from datetime import timedelta
from pathlib import Path

import earthaccess

from satchip import models


def download_data(event: models.Event, modality: models.Modality, download_path: Path) -> list[Path]:
    if not earthaccess.__auth__.authenticated:
        print('Logging in to earthaccess')
        earthaccess.login()

    results = _search_data(event, modality)

    if not results:
        return []

    local_files = earthaccess.download(results, local_path=download_path, show_progress=True)

    return local_files


def _search_data(event: models.Event, modality: models.Modality) -> list[earthaccess.DataGranule]:
    start_date = event.date
    final_date = start_date + timedelta(days=1)

    collection_id = modality['collection']

    results = earthaccess.search_data(
        short_name=[collection_id],
        temporal=(start_date.strftime('%Y-%m-%d'), final_date.strftime('%Y-%m-%d')),
        bounding_box=event.buffered_geometry().bounds,
    )

    return results

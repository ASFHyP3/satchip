from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Modality:
    id: str
    all_bands: Tuple[str, ...]
    stack_bands: Tuple[str, ...]
    chip_bands: Tuple[str, ...]

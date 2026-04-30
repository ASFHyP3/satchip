from dataclasses import dataclass


@dataclass(frozen=True)
class Modality:
    id: str
    all_bands: tuple[str, ...]
    stack_bands: tuple[str, ...]
    chip_bands: tuple[str, ...]

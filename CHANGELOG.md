# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0]

### Added
* Add mypy to static analysis github action.
* Support for RTC products.

### Changed
* Names of platforms so OPERA RTC is `S1RTC` and HyP3 RTC is `HYP3S1RTC`.

## [0.5.0]

### Changed
* Format and layout of chips to more closely match the TerraMesh dataset.

## [0.4.0]

### Changed
* For the S1RTC platform `chipdata` now processes and download all nessary RTC prior to getting individual chip data.

### Fixed
* Fix fmask path when chipping HLS data

## [0.3.0]

### Added
* Support for specifying per-chip maximum cloud cover percentage when creating Sentinel-2 and HLS chips.
* Support for getting all images within a date range or the best one.

### Changed
* `chipdata` interface so that a date range must be provided.
* Zarr structure for image chip datasets to support multi-temporal chips.
* HLS and S2L2A band names to be capitalized color names (ex. `BLUE` instead of `B02`).

## [0.2.0]

### Added
* Support for Harmonized Landsat Sentinel-2 data.
* Support for persistent scratch directory for image downloads.
* `chipview` CLI tool for view chip datasets.

### Changed
* Alignment of TerraMind chips to be centered within MajorTom chips.

## [0.1.0]

### Added
- Ability to chip label images to MajorTom/TerraMind grids.
- Ability to create Sentinel-2 and Sentinel-1 RTC chips based on an input label chip.

## [0.0.0]

### Added
- satchip package created with the [HyP3 Cookiecutter](https://github.com/ASFHyP3/hyp3-cookiecutter)

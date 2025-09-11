"""Preprocessing pipeline for eye-tracking data.

Modules:
- io_utils: lightweight YAML + CSV loading helpers
- cleaning: sample fusion, interpolation, smoothing, pupil processing
- fixations: I-DT detection + merging
- saccades: derive saccades between fixations
- visuals: heatmaps and scanpaths
- summaries: aggregation utilities

CLI entrypoint: pipeline.py
"""

__all__ = [
    "io_utils",
    "cleaning",
    "fixations",
    "saccades",
    "visuals",
    "summaries",
]


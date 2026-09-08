"""Cozmo AI floor-plan reconstruction pipeline."""

__version__ = "0.1.0"

# Version of the output contract emitted by pipeline.run. Bump this whenever
# schema.py changes in a way a consumer would notice.
SCHEMA_VERSION = "1.0.0"

# Version stamped into every run_manifest.json. Distinct from __version__ so a
# packaging bump does not silently invalidate a benchmark comparison.
PIPELINE_VERSION = "0.1.0-skeleton"

__all__ = ["__version__", "SCHEMA_VERSION", "PIPELINE_VERSION"]

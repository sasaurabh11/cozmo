from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def ground_truth_csv() -> Path:
    return FIXTURES / "benchmark" / "ground_truth.csv"


@pytest.fixture
def results_dir() -> Path:
    return FIXTURES / "benchmark" / "results"


@pytest.fixture
def photo_capture() -> Path:
    return FIXTURES / "captures" / "demo_photo"


@pytest.fixture
def lidar_capture() -> Path:
    return FIXTURES / "captures" / "demo_lidar"

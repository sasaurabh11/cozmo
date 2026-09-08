from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
CAPTURES = FIXTURES / "captures"
STAMP = CAPTURES / ".generated"


def _generator_fingerprint() -> str:
    """Hash of the two generators, so a change to either forces a rebuild."""
    digest = hashlib.sha256()
    for name in ("generate.py", "synthesize.py"):
        digest.update((FIXTURES / name).read_bytes())
    return digest.hexdigest()


@pytest.fixture(scope="session", autouse=True)
def capture_fixtures() -> Path:
    """Build the binary capture fixtures if they are missing or stale.

    Depth frames, confidence masks and MP4s are generated, not authored: 472 KB
    across 72 binary files that a committed 0.25 s script reproduces exactly.
    Keeping them out of the history keeps the diffs readable and stops the
    fixtures drifting from the code that defines them -- a stale committed
    capture is a test that passes against last week's geometry.

    Text fixtures (ground truth, the stub-tier plan.json files) stay committed:
    they are small, they diff, and they are meant to be read in review.
    """
    fingerprint = _generator_fingerprint()
    if STAMP.is_file() and STAMP.read_text().strip() == fingerprint:
        return CAPTURES

    from tests.fixtures import generate, synthesize

    generate.write_photo_capture()
    synthesize.write_capture("synthetic_room", drop_ceiling=False)
    synthesize.write_capture("synthetic_no_ceiling", drop_ceiling=True)
    synthesize.write_ground_truth()

    CAPTURES.mkdir(parents=True, exist_ok=True)
    STAMP.write_text(fingerprint + "\n")
    return CAPTURES


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
    """Synthetic room with the ceiling observed. Geometry exact by construction."""
    return FIXTURES / "captures" / "synthetic_room"


@pytest.fixture
def lidar_capture_no_ceiling() -> Path:
    """The same room with ceiling returns dropped, as when nobody looks up."""
    return FIXTURES / "captures" / "synthetic_no_ceiling"


@pytest.fixture
def synthetic_truth() -> dict:
    """The constants tests/fixtures/synthesize.py rendered the room from."""
    from tests.fixtures import synthesize

    return {
        "width": synthesize.ROOM_WIDTH,
        "depth": synthesize.ROOM_DEPTH,
        "ceiling": synthesize.CEILING_HEIGHT,
        "floor_area": synthesize.ROOM_WIDTH * synthesize.ROOM_DEPTH,
        "door_width": synthesize.DOOR["u1"] - synthesize.DOOR["u0"],
        "window_width": synthesize.WINDOW["u1"] - synthesize.WINDOW["u0"],
        "window_sill": synthesize.WINDOW["v0"],
        "window_height": synthesize.WINDOW["v1"] - synthesize.WINDOW["v0"],
    }

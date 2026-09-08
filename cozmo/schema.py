"""The output contract.

One rule runs through this file: **every physical quantity is a Measurement,
never a bare float.** A number without an interval is a claim we cannot defend
at the walk-in test, and the scorer treats a missing interval as a failed
calibration row rather than a free pass.

Counts, indices, ids, unitless confidences and 2D coordinates in a solved frame
are not physical quantities in that sense and stay plain.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import SCHEMA_VERSION


class StrictModel(BaseModel):
    """Base for every contract object: unknown keys are an error, not a shrug."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class Tier(str, Enum):
    """Input tier. Read from capture.json, never passed as a CLI flag."""

    PHOTO = "photo"
    VIDEO = "video"
    LIDAR = "lidar"


class Unit(str, Enum):
    METERS = "m"
    SQUARE_METERS = "m2"
    CUBIC_METERS = "m3"
    LINEAR_FEET = "lf"
    SQUARE_FEET = "sf"
    EACH = "ea"
    HOURS = "hr"
    RATIO = "ratio"


class OpeningType(str, Enum):
    DOOR = "door"
    WINDOW = "window"
    PASS_THROUGH = "pass_through"
    ARCHWAY = "archway"
    CLOSET = "closet"


class SurfaceKind(str, Enum):
    WALL = "wall"
    FLOOR = "floor"
    CEILING = "ceiling"


class DamageClass(str, Enum):
    WATER = "water"
    MOLD = "mold"
    FIRE_SMOKE = "fire_smoke"
    IMPACT = "impact"
    CRACK = "crack"
    STAIN = "stain"
    MISSING_MATERIAL = "missing_material"


class ScaleSource(str, Enum):
    """How metric scale entered the reconstruction.

    The photo tier has no metric sensor, so scale is recovered from a prior or a
    reference object; that is the dominant term in its error budget and the
    reason its intervals are wider.
    """

    LIDAR_DEPTH = "lidar_depth"
    ARKIT_VIO = "arkit_vio"
    REFERENCE_OBJECT = "reference_object"
    STRUCTURAL_PRIOR = "structural_prior"
    USER_SUPPLIED = "user_supplied"


class DriftMethod(str, Enum):
    """What we do about accumulated drift. POSES_AS_IS is an automatic fail on
    the drift-accountability row and exists only so the ablation can express it."""

    NONE_POSES_AS_IS = "poses_as_is"
    LOOP_CLOSURE = "loop_closure"
    POSE_GRAPH = "pose_graph"
    PLANE_ANCHORED = "plane_anchored"
    MANHATTAN_SNAP = "manhattan_snap"


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


class Measurement(StrictModel):
    """A value with a 95% confidence interval.

    ``ci_95`` is an absolute interval in the same unit as ``value`` -- not a
    plus/minus, not a percentage -- so a consumer never has to guess.
    """

    value: float
    ci_95: Tuple[float, float]
    unit: Unit = Unit.METERS

    @field_validator("ci_95")
    @classmethod
    def _ordered(cls, ci: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = ci
        if lo > hi:
            raise ValueError(f"ci_95 bounds out of order: {lo} > {hi}")
        return ci

    @model_validator(mode="after")
    def _brackets_value(self) -> "Measurement":
        lo, hi = self.ci_95
        if not (lo <= self.value <= hi):
            raise ValueError(f"ci_95 {self.ci_95} does not bracket value {self.value}")
        return self

    @property
    def half_width(self) -> float:
        lo, hi = self.ci_95
        return (hi - lo) / 2.0

    @property
    def relative_half_width(self) -> Optional[float]:
        return self.half_width / abs(self.value) if self.value else None

    def contains(self, truth: float) -> bool:
        """Does the interval cover a ground-truth value? This is the calibration
        question the benchmark asks of every measurement.

        The 1e-9 slack keeps a truth value sitting exactly on an interval bound
        from being ruled outside it by binary floating point.
        """
        lo, hi = self.ci_95
        return (lo - 1e-9) <= truth <= (hi + 1e-9)

    @classmethod
    def symmetric(cls, value: float, half_width: float, unit: Unit = Unit.METERS) -> "Measurement":
        h = abs(half_width)
        return cls(value=value, ci_95=(value - h, value + h), unit=unit)

    @classmethod
    def relative(cls, value: float, fraction: float, unit: Unit = Unit.METERS) -> "Measurement":
        """Interval as a fraction of the value -- how the photo tier widens."""
        return cls.symmetric(value, abs(value) * abs(fraction), unit)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


class Point2D(StrictModel):
    """A point in a solved metric frame (room-local or property). Plain floats:
    the uncertainty lives on the Measurements derived from these points."""

    x: float
    y: float


class Pose2D(StrictModel):
    """Placement of a room in the property frame: rotate by ``theta_rad`` then
    translate. This is what makes the stitched plan a plan and not a pile."""

    x: float
    y: float
    theta_rad: float = 0.0


class Wall(StrictModel):
    id: str
    start: Point2D
    end: Point2D
    length: Measurement
    height: Measurement
    thickness: Optional[Measurement] = None
    is_exterior: Optional[bool] = None
    opening_ids: List[str] = Field(default_factory=list)
    # Free-text note for a wall we could only partially observe -- mirrors,
    # glass and dark surfaces show up here.
    observation_note: Optional[str] = None


class Opening(StrictModel):
    id: str
    wall_id: str
    type: OpeningType
    width: Measurement
    height: Measurement
    offset_along_wall: Measurement
    sill_height: Optional[Measurement] = None
    # Room id on the far side, when we believe the opening connects two rooms.
    connects_room_id: Optional[str] = None
    detection_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class Surface(StrictModel):
    """An addressable surface. Damage regions and scope line items key to these
    ids, so a scope item can always be traced back to a place in the property."""

    id: str
    room_id: str
    kind: SurfaceKind
    wall_id: Optional[str] = None
    area: Measurement

    @model_validator(mode="after")
    def _wall_surface_has_wall(self) -> "Surface":
        if self.kind is SurfaceKind.WALL and not self.wall_id:
            raise ValueError(f"wall surface {self.id} must reference a wall_id")
        return self


class Room(StrictModel):
    id: str
    name: str
    pose: Pose2D
    walls: List[Wall]
    openings: List[Opening] = Field(default_factory=list)
    surfaces: List[Surface] = Field(default_factory=list)
    ceiling_height: Measurement
    floor_area: Measurement
    perimeter: Measurement
    # Which capture(s) contributed. Repeat captures of one room carry the same
    # room id with different capture ids -- that pairing is the repeatability gate.
    source_frame_count: Optional[int] = None

    @model_validator(mode="after")
    def _openings_reference_known_walls(self) -> "Room":
        wall_ids = {w.id for w in self.walls}
        for op in self.openings:
            if op.wall_id not in wall_ids:
                raise ValueError(f"opening {op.id} references unknown wall {op.wall_id}")
        return self


class Adjacency(StrictModel):
    """Two rooms and the reason we believe they touch."""

    room_a_id: str
    room_b_id: str
    via_opening_id: Optional[str] = None
    shared_wall_ids: Tuple[Optional[str], Optional[str]] = (None, None)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)

    @model_validator(mode="after")
    def _distinct(self) -> "Adjacency":
        if self.room_a_id == self.room_b_id:
            raise ValueError("adjacency must connect two distinct rooms")
        return self


# --------------------------------------------------------------------------
# Damage, concealment, scope
# --------------------------------------------------------------------------


class DamageRegion(StrictModel):
    id: str
    room_id: str
    surface_id: str
    damage_class: DamageClass
    # Polygon in surface-local metric coordinates (origin at the surface's
    # lower-left when facing it).
    polygon: List[Point2D] = Field(default_factory=list)
    area: Measurement
    max_extent: Measurement
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: List[str] = Field(default_factory=list)

    @field_validator("polygon")
    @classmethod
    def _polygon_is_a_polygon(cls, pts: List[Point2D]) -> List[Point2D]:
        if pts and len(pts) < 3:
            raise ValueError("polygon needs at least 3 points when present")
        return pts


class ConcealedFlag(StrictModel):
    """A place we believe is damaged behind a surface we cannot see.

    ``rule_id`` and ``rule_text`` are mandatory: the contract requires the rule
    that fired, not just the flag.
    """

    id: str
    room_id: str
    surface_id: str
    rule_id: str
    rule_text: str
    triggered_by_damage_ids: List[str] = Field(default_factory=list)
    probability: float = Field(ge=0.0, le=1.0)
    recommended_action: str
    inspection_priority: int = Field(ge=1, le=5, default=3)


class ScopeItem(StrictModel):
    """A line item keyed to a surface, so every cost traces to a place."""

    id: str
    room_id: str
    surface_id: str
    damage_region_ids: List[str] = Field(default_factory=list)
    code: str
    description: str
    quantity: Measurement
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Run-level metadata
# --------------------------------------------------------------------------


class ScaleInfo(StrictModel):
    source: ScaleSource
    # Multiplier applied to the reconstruction to reach metres. 1.0 when the
    # sensor was already metric.
    scale_factor: Measurement
    reference_description: Optional[str] = None
    notes: Optional[str] = None


class DriftCorrection(StrictModel):
    enabled: bool
    method: DriftMethod
    loop_closures: int = Field(ge=0, default=0)
    residual_closure_error: Optional[Measurement] = None
    # Footprint area with correction disabled, for the required ablation.
    ablation_footprint_area: Optional[Measurement] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _enabled_means_a_method(self) -> "DriftCorrection":
        if self.enabled and self.method is DriftMethod.NONE_POSES_AS_IS:
            raise ValueError("drift correction enabled but method is poses_as_is")
        return self


class PropertyTotals(StrictModel):
    room_count: int = Field(ge=0)
    total_floor_area: Measurement
    footprint_area: Measurement
    total_wall_area: Optional[Measurement] = None
    bounding_box_m: Optional[Tuple[float, float]] = None


class QualityReport(StrictModel):
    """How much to trust the numbers above, in the pipeline's own words."""

    overall_confidence: float = Field(ge=0.0, le=1.0)
    interval_method: str
    calibration_note: Optional[str] = None
    # Named degradations we detected: mirrors, glass, wet-look floor, low light.
    degradations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    coverage: Dict[str, float] = Field(default_factory=dict)


class Plan(StrictModel):
    """Top-level output. One of these per capture, written as plan.json."""

    schema_version: str = SCHEMA_VERSION
    capture_id: str
    tier: Tier
    pipeline_version: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scale: ScaleInfo
    drift_correction: DriftCorrection
    property_totals: PropertyTotals
    rooms: List[Room]
    adjacencies: List[Adjacency] = Field(default_factory=list)
    damage: List[DamageRegion] = Field(default_factory=list)
    concealed_flags: List[ConcealedFlag] = Field(default_factory=list)
    scope: List[ScopeItem] = Field(default_factory=list)
    quality: QualityReport

    # -- referential integrity -------------------------------------------------
    @model_validator(mode="after")
    def _ids_resolve(self) -> "Plan":
        room_ids = {r.id for r in self.rooms}
        if len(room_ids) != len(self.rooms):
            raise ValueError("duplicate room ids")
        surface_ids = {s.id for r in self.rooms for s in r.surfaces}
        damage_ids = {d.id for d in self.damage}

        for adj in self.adjacencies:
            for rid in (adj.room_a_id, adj.room_b_id):
                if rid not in room_ids:
                    raise ValueError(f"adjacency references unknown room {rid}")

        for dmg in self.damage:
            if dmg.room_id not in room_ids:
                raise ValueError(f"damage {dmg.id} references unknown room {dmg.room_id}")
            if surface_ids and dmg.surface_id not in surface_ids:
                raise ValueError(f"damage {dmg.id} references unknown surface {dmg.surface_id}")

        for flag in self.concealed_flags:
            for did in flag.triggered_by_damage_ids:
                if did not in damage_ids:
                    raise ValueError(f"concealed flag {flag.id} references unknown damage {did}")

        for item in self.scope:
            if surface_ids and item.surface_id not in surface_ids:
                raise ValueError(f"scope item {item.id} references unknown surface {item.surface_id}")
            for did in item.damage_region_ids:
                if did not in damage_ids:
                    raise ValueError(f"scope item {item.id} references unknown damage {did}")
        return self

    @property
    def room_by_id(self) -> Dict[str, Room]:
        return {r.id: r for r in self.rooms}

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        return cls.model_validate_json(text)


__all__ = [
    "Adjacency", "ConcealedFlag", "DamageClass", "DamageRegion", "DriftCorrection",
    "DriftMethod", "Measurement", "Opening", "OpeningType", "Plan", "Point2D",
    "Pose2D", "PropertyTotals", "QualityReport", "Room", "ScaleInfo", "ScaleSource",
    "ScopeItem", "StrictModel", "Surface", "SurfaceKind", "Tier", "Unit", "Wall",
]

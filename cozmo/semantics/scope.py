"""Damage regions to scope line items.

A CSV lookup, keyed on (damage class, surface kind). Nothing here is inferred:
the catalogue is a table a estimator can read and argue with, and every line
item carries a ``basis`` string spelling out the arithmetic that produced its
quantity, because a quantity a person cannot check is a quantity they will not
trust.

Four bases, chosen by the catalogue row:

``area``     the damaged patch, grown by the row's cut-back margin
``surface``  the whole surface -- for work that cannot stop at the damage
             (a repaint flashes; soot is not confined to the scorch mark)
``extent``   a length, for linear work like filling a crack
``count``    per occurrence
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger("cozmo.semantics.scope")

DEFAULT_CATALOGUE_PATH = Path(__file__).with_name("scope_items.csv")

VALID_BASES = {"area", "surface", "extent", "count"}


@dataclass(frozen=True)
class CatalogueRow:
    damage_class: str
    surface_kind: str
    code: str
    description: str
    unit: str
    basis: str
    cut_back_m: float
    waste_factor: float
    min_quantity: float
    notes: str = ""


@dataclass
class ScopeLine:
    """One line item, with the arithmetic that produced it."""

    code: str
    description: str
    unit: str
    quantity: float
    ci_95: Tuple[float, float]
    basis: str                      # the sentence explaining the number
    room_id: str
    surface_id: str
    damage_region_ids: List[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code, "description": self.description, "unit": self.unit,
            "quantity": round(self.quantity, 4),
            "ci_95": [round(v, 4) for v in self.ci_95],
            "basis": self.basis, "surface_id": self.surface_id,
            "damage_region_ids": list(self.damage_region_ids),
        }


class ScopeCatalogue:
    def __init__(self, rows: Sequence[CatalogueRow]) -> None:
        self.rows = list(rows)
        self._index: Dict[Tuple[str, str], List[CatalogueRow]] = {}
        for row in self.rows:
            self._index.setdefault((row.damage_class, row.surface_kind), []).append(row)

    def lookup(self, damage_class: str, surface_kind: str) -> List[CatalogueRow]:
        return list(self._index.get((damage_class, surface_kind), []))

    @property
    def damage_classes(self) -> set:
        return {row.damage_class for row in self.rows}


def load_catalogue(path: Optional[Path] = None) -> ScopeCatalogue:
    path = Path(path or DEFAULT_CATALOGUE_PATH)
    rows: List[CatalogueRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"damage_class", "surface_kind", "code", "description", "unit", "basis"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
        for line_no, raw in enumerate(reader, start=2):
            if not (raw.get("code") or "").strip():
                continue
            basis = (raw["basis"] or "").strip().lower()
            if basis not in VALID_BASES:
                raise ValueError(
                    f"{path}:{line_no} unknown basis '{basis}' (expected one of {sorted(VALID_BASES)})"
                )
            rows.append(CatalogueRow(
                damage_class=raw["damage_class"].strip(),
                surface_kind=raw["surface_kind"].strip(),
                code=raw["code"].strip(),
                description=raw["description"].strip(),
                unit=raw["unit"].strip(),
                basis=basis,
                cut_back_m=float(raw.get("cut_back_m") or 0.0),
                waste_factor=float(raw.get("waste_factor") or 0.0),
                min_quantity=float(raw.get("min_quantity") or 0.0),
                notes=(raw.get("notes") or "").strip(),
            ))
    if not rows:
        raise ValueError(f"{path} contains no catalogue rows")
    log.info("loaded %d scope catalogue rows from %s", len(rows), path.name)
    return ScopeCatalogue(rows)


def _grown_area(width_m: float, height_m: float, margin_m: float) -> float:
    """Damaged patch plus a cut-back margin on every side."""
    return max(0.0, width_m + 2 * margin_m) * max(0.0, height_m + 2 * margin_m)


def quantity_for(
    row: CatalogueRow,
    region: Mapping[str, Any],
    surface_area_m2: Optional[float] = None,
) -> Tuple[float, Tuple[float, float], str]:
    """Quantity, interval and the sentence explaining both."""
    area = float(region.get("area_m2") or 0.0)
    extent = float(region.get("max_extent_m") or 0.0)
    width = float(region.get("bbox_width_m") or extent)
    height = float(region.get("bbox_height_m") or (area / width if width else 0.0))
    area_ci = region.get("area_ci_95") or (area, area)
    margin = row.cut_back_m

    if row.basis == "area":
        base = _grown_area(width, height, margin)
        quantity = base * (1.0 + row.waste_factor)
        basis = (
            f"damaged patch {width:.2f} x {height:.2f} m, cut back {margin:.2f} m each side "
            f"-> {width + 2 * margin:.2f} x {height + 2 * margin:.2f} m = {base:.2f} m2"
            + (f", +{row.waste_factor:.0%} waste = {quantity:.2f} m2" if row.waste_factor else "")
        )
        low = _grown_area(width * _scale(area_ci[0], area), height * _scale(area_ci[0], area), margin)
        high = _grown_area(width * _scale(area_ci[1], area), height * _scale(area_ci[1], area), margin)
        interval = (low * (1 + row.waste_factor), high * (1 + row.waste_factor))

    elif row.basis == "surface":
        if surface_area_m2 is None:
            surface_area_m2 = area
            note = "surface area unknown; falling back to the damaged area"
        else:
            note = "whole surface, because this work cannot stop at the damage"
        quantity = surface_area_m2 * (1.0 + row.waste_factor)
        basis = (
            f"{note}: surface {surface_area_m2:.2f} m2"
            + (f" +{row.waste_factor:.0%} waste = {quantity:.2f} m2" if row.waste_factor else "")
        )
        interval = (quantity * 0.9, quantity * 1.1)

    elif row.basis == "extent":
        base = extent + 2 * margin
        quantity = base * (1.0 + row.waste_factor)
        basis = (
            f"extent {extent:.2f} m plus {margin:.2f} m overrun each end = {base:.2f} m"
            + (f", +{row.waste_factor:.0%} = {quantity:.2f} m" if row.waste_factor else "")
        )
        interval = (max(0.0, base - 0.1) , base + 0.1)

    else:  # count
        quantity = 1.0
        basis = "one occurrence"
        interval = (1.0, 1.0)

    if quantity < row.min_quantity:
        basis += f"; raised to the {row.min_quantity:g} {row.unit} minimum charge"
        interval = (row.min_quantity, max(interval[1], row.min_quantity))
        quantity = row.min_quantity

    low, high = min(interval), max(interval)
    return quantity, (min(low, quantity), max(high, quantity)), basis


def _scale(value: float, reference: float) -> float:
    """Linear scale factor for a bounding box implied by an area interval."""
    if reference <= 0:
        return 1.0
    return math.sqrt(max(value, 0.0) / reference)


def price_regions(
    regions: Sequence[Mapping[str, Any]],
    catalogue: Optional[ScopeCatalogue] = None,
    surface_areas: Optional[Mapping[str, float]] = None,
) -> List[ScopeLine]:
    """Turn damage regions into scope lines keyed to their surfaces."""
    catalogue = catalogue or load_catalogue()
    surface_areas = surface_areas or {}
    lines: List[ScopeLine] = []

    for region in regions:
        damage_class = str(region.get("damage_class", ""))
        surface_kind = str(region.get("surface_kind", ""))
        rows = catalogue.lookup(damage_class, surface_kind)
        if not rows:
            log.info(
                "no scope catalogue row for %s on a %s; region %s produces no line item",
                damage_class, surface_kind, region.get("region_id"),
            )
            continue

        for row in rows:
            surface_id = str(region.get("surface_id", ""))
            quantity, interval, basis = quantity_for(
                row, region, surface_areas.get(surface_id)
            )
            lines.append(ScopeLine(
                code=row.code, description=row.description, unit=row.unit,
                quantity=quantity, ci_95=interval, basis=basis,
                room_id=str(region.get("room_id", "")),
                surface_id=surface_id,
                damage_region_ids=[str(region.get("region_id", ""))],
                notes=row.notes,
            ))

    log.info("scope: %d line item(s) from %d region(s)", len(lines), len(regions))
    return lines

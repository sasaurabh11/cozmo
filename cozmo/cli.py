"""Command line entry points.

Two commands, one command per capture::

    cozmo run --input DIR --out DIR [--drift-correction on|off]
    cozmo benchmark --results DIR --ground-truth CSV --out DIR

Tier is never a flag. It is read from ``capture.json`` inside the input
directory, because the tier is a fact about the capture, and a flag would let a
photo capture be scored against LiDAR gates by typo.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from . import PIPELINE_VERSION, SCHEMA_VERSION, __version__
from .benchmark.score import render_table, score_results, write_results
from .geometry.fuse import DEFAULT_STRIDE, DEFAULT_VOXEL_M
from .pipeline.run import run_capture
from .semantics.stage import DEFAULT_FRAME_STRIDE, DEFAULT_MAX_FRAMES
from .seed import DEFAULT_SEED

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Cozmo AI floor-plan reconstruction pipeline.",
)


class OnOff(str, Enum):
    ON = "on"
    OFF = "off"

    @property
    def enabled(self) -> bool:
        return self is OnOff.ON


def _err(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)


@app.command()
def run(
    input_dir: Path = typer.Option(
        ..., "--input", "-i",
        help="Capture directory containing capture.json.",
        exists=True, file_okay=False, dir_okay=True, readable=True,
    ),
    out_dir: Path = typer.Option(
        ..., "--out", "-o", help="Output directory for plan.json and run_manifest.json.",
    ),
    drift_correction: OnOff = typer.Option(
        OnOff.ON, "--drift-correction",
        help="Drift correction on the multi-room stitch. 'off' is the ablation arm.",
        case_sensitive=False,
    ),
    seed: int = typer.Option(DEFAULT_SEED, "--seed", help="Global RNG seed, recorded in the manifest."),
    stride: int = typer.Option(
        DEFAULT_STRIDE, "--stride", min=1,
        help="LiDAR tier: use every Nth frame when fusing. Lower is denser and slower.",
    ),
    voxel_size: float = typer.Option(
        DEFAULT_VOXEL_M, "--voxel-size", min=0.0,
        help="LiDAR tier: voxel downsample size in metres. 0 disables downsampling.",
    ),
    semantics: bool = typer.Option(
        True, "--semantics/--no-semantics",
        help="Run damage detection, concealed-damage rules and scope. Needs model "
             "weights (scripts/fetch_weights.sh).",
    ),
    weights_dir: Optional[Path] = typer.Option(
        None, "--weights-dir", help="Where the model weights live. Default: ./weights."
    ),
    frame_stride: int = typer.Option(
        DEFAULT_FRAME_STRIDE, "--frame-stride", min=1,
        help="Detect on every Nth RGB frame.",
    ),
    max_frames: int = typer.Option(
        DEFAULT_MAX_FRAMES, "--max-frames", min=1,
        help="Cap on RGB frames sent to the detector.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log each reconstruction stage."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress the summary."),
) -> None:
    """Run the pipeline on one capture."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )
    try:
        result = run_capture(
            input_dir=input_dir,
            out_dir=out_dir,
            drift_correction=drift_correction.enabled,
            seed=seed,
            command=["cozmo", *sys.argv[1:]],
            stride=stride,
            voxel_size_m=voxel_size,
            semantics=semantics,
            weights_dir=weights_dir,
            frame_stride=frame_stride,
            max_frames=max_frames,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        _err(f"run failed: {exc}")
        raise typer.Exit(code=2)

    if quiet:
        return

    plan = result.plan
    typer.echo(f"capture      {plan.capture_id}")
    typer.echo(f"tier         {plan.tier.value}   (from capture.json)")
    typer.echo(f"rooms        {len(plan.rooms)}   adjacencies {len(plan.adjacencies)}")
    typer.echo(
        f"footprint    {plan.property_totals.footprint_area.value:.2f} m2 "
        f"[{plan.property_totals.footprint_area.ci_95[0]:.2f}, "
        f"{plan.property_totals.footprint_area.ci_95[1]:.2f}]"
    )
    typer.echo(
        f"drift        {plan.drift_correction.method.value} "
        f"({'on' if plan.drift_correction.enabled else 'off'})"
    )
    room = plan.rooms[0] if plan.rooms else None
    if room is not None:
        typer.echo(
            f"ceiling      {room.ceiling_height.value:.2f} m "
            f"[{room.ceiling_height.ci_95[0]:.2f}, {room.ceiling_height.ci_95[1]:.2f}] "
            f"({plan.quality.ceiling_method or 'n/a'})"
        )
        walls = ", ".join(f"{w.length.value:.2f}" for w in room.walls[:8])
        typer.echo(f"walls        {len(room.walls)}: {walls}{' ...' if len(room.walls) > 8 else ''} m")
        typer.echo(f"openings     {len(room.openings)}")
    typer.echo(
        f"damage       {len(plan.damage)} region(s), {len(plan.concealed_flags)} concealed flag(s), "
        f"{len(plan.scope)} scope line(s)"
    )
    for region in plan.damage:
        typer.echo(
            f"  - {region.damage_class.value:<16} {region.area.value:.2f} m2 "
            f"[{region.area.ci_95[0]:.2f}, {region.area.ci_95[1]:.2f}] on {region.surface_id}"
        )
    for flag in plan.concealed_flags:
        typer.echo(f"  ! {flag.rule_id:<26} p={flag.probability:.2f}  {flag.triggering_values}")
    typer.echo(f"plan         {result.plan_path}")
    typer.echo(f"manifest     {result.manifest_path}")
    for path in result.rendered:
        typer.echo(f"drawing      {path}")
    for warning in plan.quality.warnings:
        typer.secho(f"warning      {warning}", fg=typer.colors.YELLOW)


@app.command()
def benchmark(
    results_dir: Path = typer.Option(
        ..., "--results", "-r",
        help="Directory searched recursively for plan.json files (or a single plan.json).",
        exists=True, readable=True,
    ),
    ground_truth: Path = typer.Option(
        ..., "--ground-truth", "-g",
        help="Ground-truth CSV: room,element,element_id,dimension,value_m,method,notes",
        exists=True, dir_okay=False, readable=True,
    ),
    out_dir: Path = typer.Option(..., "--out", "-o", help="Where results.json is written."),
    strict: bool = typer.Option(
        False, "--strict", help="Exit non-zero when any gate fails. Off by default: the table is a report."
    ),
) -> None:
    """Score plans against ground truth and print the gate table."""
    try:
        report = score_results(results_dir=results_dir, ground_truth_csv=ground_truth)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        _err(f"benchmark failed: {exc}")
        raise typer.Exit(code=2)

    typer.echo(render_table(report))
    path = write_results(report, out_dir)
    typer.echo(f"\nresults      {path}")

    if strict and report.failed:
        raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Print package, pipeline and schema versions."""
    typer.echo(f"cozmo {__version__}")
    typer.echo(f"pipeline {PIPELINE_VERSION}")
    typer.echo(f"schema {SCHEMA_VERSION}")


def main(argv: Optional[list] = None) -> None:  # pragma: no cover - thin wrapper
    app(args=argv)


if __name__ == "__main__":  # pragma: no cover
    app()

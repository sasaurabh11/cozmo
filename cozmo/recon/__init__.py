"""Photo-tier reconstruction: backbone, scale recovery, layout, frames.

Tier-agnostic like ``cozmo.semantics``: everything here takes ``Frame`` objects
and a ``Reconstructor``, and knows nothing about capture.json or the CLI.

**No eager re-exports.** ``cozmo.recon.layout`` and ``cozmo.recon.scale`` import
``cozmo.geometry`` (for the LiDAR-path fallback and shared polygon assembly),
which imports open3d; ``cozmo.recon.worker`` runs under ``.venv-recon``
alongside torch, and open3d + torch cannot share a process (see
``cozmo.semantics.worker`` for the same conflict with a different pair of
libraries). If this package's ``__init__`` imported everything at package load,
``python -m cozmo.recon.worker`` would drag open3d into that process the moment
it imported ``cozmo.recon.backbone`` -- since importing any submodule runs the
parent package's ``__init__`` first. So callers import what they need directly:
``from cozmo.recon.backbone import ...``, ``from cozmo.recon.layout import ...``,
etc. -- exactly as the rest of this codebase already does.
"""

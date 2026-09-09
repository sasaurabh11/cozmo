"""Multi-room stitching: pose graph over rooms, one connected property plan.

Tier-agnostic in the same sense as ``cozmo.recon`` and ``cozmo.semantics``: the
matching and graph-optimisation code operates on room reconstructions (a floor
frame, a polygon, openings, and a handful of representative frames with known
intrinsics) rather than on raw sensor data, so it does not care whether a room
came from LiDAR or from photos.

**No eager re-exports**, for the reason documented in ``cozmo.recon``:
``cozmo.stitch.match`` needs torch (DINOv2, SuperPoint, LightGlue) and runs in
its own process (see ``cozmo.stitch.worker``); ``cozmo.stitch.graph`` needs
open3d for pose-graph optimisation. If this package's ``__init__`` imported
both eagerly, the worker process would drag open3d in the moment it imported
anything from this package. Import the submodule you need directly.
"""

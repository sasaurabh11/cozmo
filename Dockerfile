# Runs the CLI on a clean machine with no local Python setup:
#
#   docker build -t cozmo .
#   docker run --rm -v "$PWD/capture:/data/capture" -v "$PWD/out:/data/out" \
#          cozmo run --input /data/capture --out /data/out
#
# Weights are fetched by script into a volume, never baked into the image.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgl/libgomp are open3d and opencv runtime dependencies; the slim image has
# neither, and both fail at import rather than at install without them.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates git libgl1 libgomp1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependency layer first, so source edits do not re-resolve the world.
COPY pyproject.toml README.md ./
COPY cozmo/__init__.py cozmo/__init__.py
RUN pip install --no-cache-dir .

COPY cozmo ./cozmo
COPY tests ./tests
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-deps -e . \
 && chmod +x scripts/*.sh \
 && ./scripts/fetch_weights.sh /weights

ENV COZMO_WEIGHTS_DIR=/weights
VOLUME ["/data"]

ENTRYPOINT ["cozmo"]
CMD ["--help"]

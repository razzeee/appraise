# syntax=docker/dockerfile:1
# Use the same Debian release as the backend. Python, virtualenv shebangs and
# the deployment manifest require the absolute /opt paths.
FROM debian:testing-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/appraise-python HF_HOME=/opt/appraise-hf
RUN uv python install 3.12
COPY runtime/release.json /tmp/release.json
RUN git clone https://github.com/razzeee/appraise.git /opt/appraise && \
    git -C /opt/appraise checkout "$(uv run --no-project --python 3.12 python -c 'import json; print(json.load(open("/tmp/release.json"))["commit"])')"
WORKDIR /opt/appraise
RUN uv sync --python 3.12 --extra embeddings --locked --no-dev --no-cache
ADD --checksum=sha256:5e9814e4604b5eff5c3ad2d2b1a5e940d46135b85ef56c5d3790757ae235551e https://github.com/razzeee/appraise/releases/download/v0.2.2/appraise-models-v0.2.2.tar.gz /tmp/models.tar.gz
COPY runtime/prepare.py /tmp/prepare.py
RUN .venv/bin/python /tmp/prepare.py && rm -rf .git /tmp/models.tar.gz

FROM debian:testing-slim AS verify
COPY --from=build /opt /opt
ENV HF_HOME=/opt/appraise-hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /opt/appraise
# Exercise the exported environment in a fresh image with networking disabled.
RUN --network=none .venv/bin/python -c 'from PIL import Image; Image.new("RGBA", (128, 128), "blue").save("/tmp/icon.png")' && \
    .venv/bin/appraise review examples/listing.json --icon /tmp/icon.png \
      --model artifacts/listing/model.joblib --icon-model artifacts/calibrated-clip/model.joblib > /tmp/review.json && \
    .venv/bin/appraise review examples/listing.json --icon /tmp/icon.png \
      --model artifacts/listing/model.joblib --icon-model artifacts/calibrated-clip/model.joblib \
      --format moderation-json > /tmp/moderation.json
RUN mkdir /out && tar -czf /out/appraise-runtime-linux-amd64.tar.gz -C / opt

FROM scratch AS export
COPY --from=verify /out/ /

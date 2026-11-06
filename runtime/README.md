# Offline backend runtime

Appraise CI builds the Python environment and downloads the model files and
CLIP weights. The website backend downloads and extracts this archive.

The archive includes the locked Python environment, a managed Python 3.12,
the complete trained model bundle, the pinned CLIP encoder cache and
`/opt/appraise/deployment.json`. It uses the existing evaluated models.

## Publish

Run the `Publish offline runtime` workflow on the source revision you want to
release. Use a new tag, starting with `runtime-v0.2.2-1`.
CI builds for linux/amd64 and tests combined review and moderation export in a
fresh Debian image with networking disabled. It publishes these files:

- `appraise-runtime-linux-amd64.tar.gz`
- `release.json`, which records the model version, source commit and encoder revision
- `SHA256SUMS`

Use separate tags for runtime releases and Python package releases. Never reuse
a runtime tag. To update models, update `release.json` and the model URL and
checksum in `Dockerfile` together. The build verifies that both checksums agree.
Model releases still need all the files listed in `PRODUCTION.md`.

To build locally, use Docker BuildKit:

```sh
docker buildx build --platform linux/amd64 -f runtime/Dockerfile \
  --output type=local,dest=dist/runtime .
```

## Backend setup

Use a Debian testing linux/amd64 image. Extract the archive at `/` and preserve
the `/opt` paths and symlinks. Python and the virtual environment depend on those
paths. Keep the backend's expected model identity in sync with `release.json`.
The backend verifies the manifest and model hashes before inference.
Set `HF_HOME=/opt/appraise-hf`,
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

Pin the runtime tag in the Dockerfile. Set `APPRAISE_RUNTIME_SHA256` to the
archive's digest from the release's `SHA256SUMS` file. Docker's remote `ADD`
verifies that digest. A separate `tar` command extracts the archive.
The website's `Dockerfile.appraise` uses these steps. Compose passes the
checksum environment variable as a build argument.

```sh
export APPRAISE_RUNTIME_SHA256=<published-archive-sha256>
docker compose --profile appraise up --build
```

Publish the first runtime release before building the backend image.
The backend requires the pinned archive and its checksum.

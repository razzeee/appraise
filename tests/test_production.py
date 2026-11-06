import gzip
import io

import httpx
import joblib
import numpy as np
import pytest
from PIL import Image

from flathub_icon_review.artifacts import atomic_dump, load_model
from flathub_icon_review.dataset import Client, write_json
from flathub_icon_review.features import open_rgba


def client_with_transport(handler, **kwargs):
    client = Client(0, **kwargs)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handler))
    return client


def test_download_size_limit():
    client = client_with_transport(
        lambda _: httpx.Response(200, content=b"x" * 33), max_bytes=32
    )
    try:
        with pytest.raises(ValueError, match="byte limit"):
            client.get("https://flathub.org/icon")
    finally:
        client.http.close()


def test_compressed_response_is_decoded_once():
    client = client_with_transport(
        lambda _: httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            content=gzip.compress(b'{"name":"Example"}'),
        )
    )
    try:
        assert client.get("https://flathub.org/app").json() == {"name": "Example"}
    finally:
        client.http.close()


def test_download_limit_applies_to_decoded_bytes():
    client = client_with_transport(
        lambda _: httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            content=gzip.compress(b"x" * 4096),
        ),
        max_bytes=1024,
    )
    try:
        with pytest.raises(ValueError, match="byte limit"):
            client.get("https://flathub.org/icon")
    finally:
        client.http.close()


@pytest.mark.parametrize(
    "target",
    [
        "https://127.0.0.1/private",
        "http://dl.flathub.org/icon",
        "https://example.org/icon",
        "https://user@flathub.org/icon",
    ],
)
def test_live_policy_checks_redirect_destinations(target):
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": target})

    client = client_with_transport(
        handle, allowed_hosts={"flathub.org", "dl.flathub.org"}
    )
    try:
        with pytest.raises(ValueError, match="trusted Flathub"):
            client.get("https://flathub.org/icon")
        assert calls == ["https://flathub.org/icon"]
    finally:
        client.http.close()


def test_allowed_redirect_and_json():
    def handle(request):
        if request.url.host == "flathub.org":
            return httpx.Response(
                302, headers={"Location": "https://dl.flathub.org/icon"}
            )
        return httpx.Response(200, json={"name": "Example"})

    client = client_with_transport(
        handle, allowed_hosts={"flathub.org", "dl.flathub.org"}
    )
    try:
        assert client.get("https://flathub.org/icon").json() == {"name": "Example"}
    finally:
        client.http.close()


def test_redirect_loop_is_bounded():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "/again"})

    client = client_with_transport(handle)
    try:
        with pytest.raises(ValueError, match="Too many"):
            client.get("https://flathub.org/icon")
        assert len(calls) == 6
    finally:
        client.http.close()


def test_atomic_model_write_preserves_previous_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    atomic_dump({"original": True}, path)
    previous = path.read_bytes()

    def fail(value, file):
        file.write(b"partial pickle")
        raise OSError("disk full")

    monkeypatch.setattr(joblib, "dump", fail)
    with pytest.raises(OSError, match="disk full"):
        atomic_dump({"new": True}, path)
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("kind", ["icon", "listing"])
@pytest.mark.parametrize("value", [[], None, {"models": {"broken": 123}}])
def test_invalid_artifact_schema_is_reported(tmp_path, kind, value):
    path = tmp_path / "model.joblib"
    joblib.dump(value, path)
    with pytest.raises(ValueError):
        load_model(path, kind)


def test_oversized_image_is_rejected_before_conversion():
    image = Image.new("1", (4097, 4097))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    with pytest.raises(ValueError, match="16-megapixel"):
        open_rgba(buffer)


def test_atomic_model_round_trip(tmp_path):
    path = tmp_path / "model.joblib"
    atomic_dump({"values": np.arange(4)}, path)
    np.testing.assert_array_equal(joblib.load(path)["values"], np.arange(4))


def test_invalid_json_never_replaces_previous_output(tmp_path):
    path = tmp_path / "evaluation.json"
    write_json(path, {"valid": True})
    previous = path.read_bytes()
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_truncated_model_has_actionable_error(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"\x80\x04\x95")
    with pytest.raises(ValueError, match="Unreadable model artifact"):
        load_model(path, "icon")

import io

import httpx
import pytest
from PIL import Image

from flathub_icon_review import flathub
from flathub_icon_review.dataset import CRITERIA, Client


@pytest.mark.parametrize(
    "url",
    [
        "https://flathub.org/apps/org.gnome.Calculator",
        "https://flathub.org/en/apps/org.gnome.Calculator/",
        "https://flathub.org/pt-BR/apps/org.gnome.Calculator?source=test#details",
        "https://www.flathub.org/apps/details/org.gnome.Calculator",
    ],
)
def test_app_page_urls(url):
    assert flathub.app_id_from_url(url) == "org.gnome.Calculator"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/apps/org.gnome.Calculator",
        "https://flathub.org.evil.example/apps/org.gnome.Calculator",
        "https://user@flathub.org/apps/org.gnome.Calculator",
        "https://flathub.org:123/apps/org.gnome.Calculator",
        "https://flathub.org/apps/org.gnome.Calculator%2Fextra",
        "https://flathub.org/apps/org.gnome.Calculator/extra",
        "https://flathub.org/apps",
        "https://flathub.org/",
        "http://flathub.org/apps/org.gnome.Calculator",
    ],
)
def test_reject_non_app_urls(url):
    with pytest.raises(ValueError):
        flathub.app_id_from_url(url)


def test_live_fetch_uses_current_api_inputs_and_cleans_up(tmp_path, monkeypatch):
    requests, icons = [], []
    app = {
        "name": "Current Name",
        "summary": "Current summary",
        "branding": [],
        "icon": "https://dl.flathub.org/icon.png",
    }
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")

    def handle(request):
        requests.append(str(request.url))
        if request.url.host == "flathub.org":
            return httpx.Response(200, json=app)
        return httpx.Response(200, content=buffer.getvalue())

    client = Client(0)
    client.http.close()
    client.http = httpx.Client(transport=httpx.MockTransport(handle))
    monkeypatch.setattr(flathub, "Client", lambda _: client)
    listing_model, icon_model = tmp_path / "listing.joblib", tmp_path / "icon.joblib"
    listing_model.touch()
    icon_model.touch()

    def review(metadata, model, icon, image_model):
        assert metadata == app
        assert model == listing_model and image_model == icon_model
        assert icon.read_bytes() == buffer.getvalue()
        icons.append(icon)
        return {"criteria": {"checked": True}}

    monkeypatch.setattr(flathub, "review_listing", review)
    result = flathub.review_url(
        "https://flathub.org/de/apps/org.gnome.Calculator", listing_model, icon_model
    )
    assert requests == [
        "https://flathub.org/api/v2/appstream/org.gnome.Calculator",
        app["icon"],
    ]
    assert result["metadata"]["name"] == "Current Name"
    assert result["source"]["app_id"] == "org.gnome.Calculator"
    assert result["source"]["fetched_at"]
    assert client.http.is_closed
    assert not icons[0].exists()


def test_missing_icon_still_reviews_text(tmp_path, monkeypatch):
    model = tmp_path / "model.joblib"
    model.touch()
    client = Client(0)
    client.http.close()
    client.http = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"name": "Example", "summary": "Read books"}
            )
        )
    )
    monkeypatch.setattr(flathub, "Client", lambda _: client)

    def review(app, listing, icon, icon_model):
        assert icon is None and icon_model is None
        return {"criteria": {"name": {"pass_score": 0.8}}}

    monkeypatch.setattr(flathub, "review_listing", review)
    result = flathub.review_url(
        "https://flathub.org/apps/org.example.App", model, model
    )
    assert result["criteria"]["name"]["pass_score"] == 0.8
    assert all("unavailable" in result["criteria"][key] for key in CRITERIA)


def test_model_preflight_and_http_failure(tmp_path, monkeypatch):
    model = tmp_path / "missing.joblib"
    url = "https://flathub.org/apps/org.example.App"
    with pytest.raises(ValueError, match="Model not found"):
        flathub.review_url(url, model, model)
    model.touch()
    client = Client(0)
    client.http.close()
    client.http = httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(404, json={"detail": "Not found"})
        )
    )
    monkeypatch.setattr(flathub, "Client", lambda _: client)
    with pytest.raises(httpx.HTTPStatusError):
        flathub.review_url(url, model, model)
    assert client.http.is_closed


def test_missing_model_hint_uses_requested_output_directory(tmp_path):
    listing = tmp_path / "listing/model.joblib"
    icon = tmp_path / "calibrated-clip/model.joblib"
    url = "https://flathub.org/apps/org.example.App"
    with pytest.raises(ValueError, match=f"train-listing --output {listing.parent}"):
        flathub.review_url(url, listing, icon)
    listing.parent.mkdir()
    listing.touch()
    with pytest.raises(
        ValueError, match=f"train --features clip --output {icon.parent}"
    ):
        flathub.review_url(url, listing, icon)

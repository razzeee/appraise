"""Review current Flathub app metadata without changing training snapshots."""

import re
import shlex
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

from .dataset import API, CRITERIA, Client
from .listing import review_listing
from .listing_features import validate_metadata


def app_id_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in ("flathub.org", "www.flathub.org")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
    ):
        raise ValueError("Use an HTTPS flathub.org app page URL")
    parts = unquote(parsed.path).strip("/").split("/")
    if parts and re.fullmatch(r"[a-z]{2,3}(?:[-_][A-Za-z]{2,4})?", parts[0]):
        parts = parts[1:]
    if len(parts) == 3 and parts[:2] == ["apps", "details"]:
        parts = ["apps", parts[2]]
    if (
        len(parts) != 2
        or parts[0] != "apps"
        or not re.fullmatch(r"[A-Za-z_][\w.\-]+", parts[1], flags=re.ASCII)
    ):
        raise ValueError(
            "Expected a Flathub app URL such as https://flathub.org/apps/org.gnome.Calculator"
        )
    return parts[1]


def review_url(url: str, model: Path, icon_model: Path):
    app_id = app_id_from_url(url)
    for path, command in (
        (model, "train-listing"),
        (icon_model, "train --features clip"),
    ):
        if not path.is_file():
            raise ValueError(
                f"Model not found: {path}. Copy the trained artifact here, "
                f"run icon-review {command} --output {shlex.quote(str(path.parent))} "
                "(writes model.joblib there), or pass an explicit model path."
            )
    client = Client(0.25)
    api_url = f"{API}/appstream/{app_id}"
    try:
        app = client.get(api_url).json()
        validate_metadata(app)
        icon_url = app.get("icon")
        if icon_url and (
            not isinstance(icon_url, str)
            or urlsplit(icon_url).scheme not in ("http", "https")
            or not urlsplit(icon_url).hostname
        ):
            raise ValueError("Flathub returned an invalid icon URL")
        temporary_root = Path("/tmp/opencode")
        with TemporaryDirectory(
            prefix="icon-taste-",
            dir=temporary_root if temporary_root.is_dir() else None,
        ) as directory:
            icon = None
            if icon_url:
                icon = Path(directory) / "icon"
                icon.write_bytes(client.get(icon_url).content)
            result = review_listing(app, model, icon, icon_model if icon else None)
            if icon is None:
                result["criteria"].update(
                    {
                        key: {"unavailable": "Flathub metadata has no icon"}
                        for key in CRITERIA
                    }
                )
        result["source"] = {
            "app_id": app_id,
            "page_url": f"https://flathub.org/apps/{app_id}",
            "api_url": api_url,
            "icon_url": icon_url,
            "fetched_at": datetime.now(UTC).isoformat(),
            "language": "API default (English); page locale is ignored",
        }
        result["metadata"] = {
            key: app.get(key) for key in ("name", "summary", "branding")
        }
        return result
    finally:
        client.http.close()

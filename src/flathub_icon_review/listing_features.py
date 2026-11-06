"""Versioned, local text and brand-color measurements; no review labels as inputs."""

import colorsys
import re
import unicodedata
from pathlib import Path

import numpy as np

from .features import open_rgba

VERSION = 1
NAME_CRITERIA = ("app-name-just-the-name", "app-no-weird-formatting")
SUMMARY_CRITERIA = (
    "app-summary-not-technical",
    "app-summary-no-weird-formatting",
    "app-summary-dont-repeat-app-name",
    "app-summary-dont-start-with-an-article",
)
BRAND_CRITERIA = ("branding-good-primary-brand-colors",)
RULE_CRITERIA = (
    "app-name-not-too-long",
    "app-summary-not-too-long",
    "branding-has-primary-brand-colors",
)
LEARNED_CRITERIA = NAME_CRITERIA + SUMMARY_CRITERIA + BRAND_CRITERIA
LISTING_CRITERIA = RULE_CRITERIA + LEARNED_CRITERIA


def category(key):
    if key in NAME_CRITERIA or key == RULE_CRITERIA[0]:
        return "name"
    if key in SUMMARY_CRITERIA or key == RULE_CRITERIA[1]:
        return "summary"
    return "branding"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def validate_metadata(app):
    if not isinstance(app, dict):
        raise ValueError("Metadata must be an AppStream JSON object")
    for field in ("name", "summary"):
        if field in app and not isinstance(app[field], str):
            raise ValueError(f"{field} must be a string (use English metadata)")
    branding = app.get("branding")
    if branding is not None and (
        not isinstance(branding, list)
        or any(not isinstance(item, dict) for item in branding)
    ):
        raise ValueError("branding must be a list of color objects or null")


def primary_colors(app):
    """Resolve explicit theme colors before a theme-independent fallback."""
    values = {}
    for item in app.get("branding") or []:
        if item.get("type") == "primary" and "value" in item:
            scheme = item.get("scheme_preference", "default")
            if scheme in ("light", "dark", "default"):
                if scheme in values:
                    raise ValueError(f"Duplicate primary color for {scheme}")
                values[scheme] = item["value"]
    result = {}
    for scheme in ("light", "dark"):
        value = values.get(scheme, values.get("default"))
        if value is None:
            continue
        if not isinstance(value, str) or not re.fullmatch(
            r"#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}", value
        ):
            raise ValueError(f"Invalid {scheme} primary color: {value!r}")
        hex_value = value[1:]
        if len(hex_value) == 3:
            hex_value = "".join(c * 2 for c in hex_value)
        result[scheme] = np.array(
            [int(hex_value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        )
    return result


def text_measurements(app, field):
    text = app[field]
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    letters = [c for c in text if c.isalpha()]
    name = normalized(app.get("name", ""))
    summary = normalized(app.get("summary", ""))
    features = {
        "characters": len(text),
        "words": len(words),
        "uppercase_fraction": sum(c.isupper() for c in letters) / max(1, len(letters)),
        "title_word_fraction": sum(w.istitle() for w in words) / max(1, len(words)),
        "lowercase": float(text.islower()),
        "uppercase": float(text.isupper()),
        "camel_case": float(bool(re.search(r"[a-z][A-Z]", text))),
        "punctuation": sum(unicodedata.category(c).startswith("P") for c in text),
        "ends_full_stop": float(text.rstrip().endswith(".")),
        "starts_article": float(bool(re.match(r"^(a|an|the)\b", normalized(text)))),
        "separator": float(bool(re.search(r"[:()–—]| - ", text))),
        "extra_whitespace": float(text != " ".join(text.split())),
    }
    if field == "summary":
        features["repeats_name"] = float(
            bool(name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", summary))
        )
        features["name_word_overlap"] = len(
            set(re.findall(r"\w+", name)) & set(re.findall(r"\w+", summary))
        )
    return features


def luminance(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear @ np.array([0.2126, 0.7152, 0.0722])


def brand_measurements(app, icon: Path):
    colors = primary_colors(app)
    if len(colors) != 2:
        raise ValueError("Good-brand-color scoring needs light and dark primary colors")
    rgba = np.asarray(open_rgba(icon).resize((64, 64))) / 255
    visible = rgba[:, :, 3] > 0.05
    if not visible.any():
        raise ValueError("Good-brand-color scoring needs a visible icon")
    rgb = rgba[:, :, :3][visible]
    alpha = rgba[:, :, 3][visible, None]
    features = {"icon_coverage": float(visible.mean())}
    for scheme, color in colors.items():
        hue, saturation, value = colorsys.rgb_to_hsv(*color)
        features.update(
            {
                f"{scheme}_hue_sin": np.sin(hue * 2 * np.pi),
                f"{scheme}_hue_cos": np.cos(hue * 2 * np.pi),
                f"{scheme}_saturation": saturation,
                f"{scheme}_value": value,
                f"{scheme}_luminance": float(luminance(color)),
            }
        )
        composite = rgb * alpha + color * (1 - alpha)
        foreground, background = luminance(composite), luminance(color)
        contrast = (np.maximum(foreground, background) + 0.05) / (
            np.minimum(foreground, background) + 0.05
        )
        for quantile in (10, 50, 90):
            features[f"{scheme}_contrast_p{quantile}"] = float(
                np.percentile(contrast, quantile)
            )
        features[f"{scheme}_low_contrast_fraction"] = float((contrast < 1.5).mean())
        features[f"{scheme}_icon_distance"] = float(
            np.average(np.linalg.norm(rgb - color, axis=1), weights=alpha[:, 0])
        )
    features["theme_distance"] = float(np.linalg.norm(colors["light"] - colors["dark"]))
    features["light_minus_dark_luminance"] = (
        features["light_luminance"] - features["dark_luminance"]
    )
    return features


def inputs(app, icon=None):
    """Return usable category inputs plus explicit reasons for unavailable ones."""
    validate_metadata(app)
    prepared, unavailable = {}, {}
    for field in ("name", "summary"):
        if not app.get(field, "").strip():
            unavailable[field] = f"Missing or empty {field}"
        elif field == "summary" and not app.get("name", "").strip():
            unavailable[field] = "Summary scoring also requires the app name"
        else:
            prepared[field] = {
                "text": app[field],
                "numeric": text_measurements(app, field),
            }
    if icon is None:
        unavailable["branding"] = "Good-brand-color scoring requires a local icon"
    else:
        try:
            prepared["branding"] = {"numeric": brand_measurements(app, icon)}
        except (ValueError, OSError) as error:
            unavailable["branding"] = str(error)
    return prepared, unavailable


def rules(app):
    """Match Flathub's automatic checks; recommendations are separate hints."""
    validate_metadata(app)
    results = {}
    for field, maximum in (("name", 20), ("summary", 35)):
        results[f"app-{field}-not-too-long"] = {
            "method": "rule",
            "passed": field in app and len(app[field]) <= maximum,
            "characters": len(app[field]) if field in app else None,
            "maximum": maximum,
        }
    results["branding-has-primary-brand-colors"] = {
        "method": "rule",
        "passed": any(
            item.get("type") == "primary" and "value" in item
            for item in app.get("branding") or []
        ),
    }
    return results


def hints(app):
    """Explain observable issues without pretending heuristics are human labels."""
    result = []
    for field in ("name", "summary"):
        if not app.get(field, "").strip():
            result.append({"field": field, "message": "Provide nonempty text"})
            continue
        measurements = text_measurements(app, field)
        checks = {
            "extra_whitespace": "Remove extra whitespace",
            "uppercase": "Check uppercase formatting; established brands may be exempt",
        }
        if field == "summary":
            checks.update(
                {
                    "ends_full_stop": "Avoid a trailing full stop",
                    "starts_article": "Prefer starting with an action, not an article",
                    "repeats_name": "Avoid repeating the app name in the summary",
                }
            )
        elif len(app[field]) > 15:
            result.append(
                {"field": field, "message": "Prefer a name of 15 characters or fewer"}
            )
        for key, message in checks.items():
            if measurements[key]:
                result.append({"field": field, "message": message})
    try:
        colors = primary_colors(app)
        if len(colors) != 2:
            result.append(
                {"field": "branding", "message": "Provide both light and dark colors"}
            )
        elif np.array_equal(colors["light"], colors["dark"]):
            result.append(
                {"field": "branding", "message": "Use distinct light and dark colors"}
            )
        elif luminance(colors["light"]) <= luminance(colors["dark"]):
            result.append(
                {
                    "field": "branding",
                    "message": "The dark color should usually be darker",
                }
            )
    except ValueError as error:
        result.append({"field": "branding", "message": str(error)})
    return result


def select_text(rows):
    return [row["text"] for row in rows]


def select_numeric(rows):
    return [row["numeric"] for row in rows]

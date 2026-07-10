"""Test the lr CLI command with all fisheye mapping types and rotated right images."""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import pytest

from vr180_convert.cli import app

_ASSETS = Path("tests/assets")
_CACHE = Path("tests/.cache")

# orthographic has limited FoV (theta must be <= 1) and produces NaN for wide-angle fisheye images
MAPPING_TYPES = ["rectilinear", "stereographic", "equidistant", "equisolid"]


@pytest.fixture(scope="module", params=[30.0, -30.0])
def rotated_path(request: pytest.FixtureRequest) -> Path:
    """Rotate 001R.JPG by ±30 degrees and save."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    img = cv.imread(str(_ASSETS / "001R.JPG"))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot = cv.getRotationMatrix2D(center, request.param, 1.0)
    rotated = cv.warpAffine(img, rot, (w, h))
    angle_str = f"{request.param:+.0f}".replace("+", "p").replace("-", "n")
    out = _CACHE / f"001R_rotated{angle_str}.jpg"
    cv.imwrite(str(out), rotated)
    return out


@pytest.mark.parametrize("mapping_type", MAPPING_TYPES)
@pytest.mark.parametrize("use_automatch", [False, True])
def test_lr_mapping_types(
    rotated_path: Path,
    mapping_type: str,
    use_automatch: bool,
) -> None:
    """Run lr with each mapping type, with and without automatch."""
    angle_str = rotated_path.stem.replace("001R_rotated", "")
    tag = f"{mapping_type}_{angle_str}_{'auto' if use_automatch else 'noauto'}"
    out = _CACHE / f"test_lr_{tag}.png"
    if out.exists():
        out.unlink()

    args = [
        str(_ASSETS / "001L.JPG"),
        str(rotated_path),
        str(out),
        "--size",
        "1024x1024",
        "--radius",
        "max",
        "--mapping-type",
        mapping_type,
    ]
    if use_automatch:
        args.append("--automatch")

    try:
        app(args)
    except SystemExit:
        pass

    assert out.exists(), f"Output {out} was not created for {tag}"
    result = cv.imread(str(out))
    assert result is not None, f"Could not read {out}"
    assert result.shape == (1024, 2048, 3), f"Unexpected shape {result.shape} for {tag}"

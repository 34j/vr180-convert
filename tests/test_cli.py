import pytest
from typer.testing import CliRunner

from vr180_convert.cli import app
from vr180_convert.testing import generate_test_image

from .test_remapper import _TEST_DIR, _TEST_IMAGE_PATH

runner = CliRunner()


@pytest.fixture(scope="session", autouse=True)
def generate_image():
    _TEST_DIR.mkdir(exist_ok=True)
    generate_test_image(2048, _TEST_IMAGE_PATH)


def test_help():
    """The help message includes the CLI name."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_lr():
    result = runner.invoke(
        app,
        [
            "lr",
            _TEST_IMAGE_PATH.as_posix(),
            _TEST_IMAGE_PATH.as_posix(),
            "--transformer",
            'FisheyeFormatEncoder("equidistant") * '
            "Euclidean3DRotator(from_rotation_vector([0, np.pi / 4, 0])) * "
            'FisheyeFormatDecoder("equidistant")',
            "--radius",
            "max",
            "--out-path",
            (_TEST_DIR / "test.cli.lr.jpg").as_posix(),
        ],
    )
    assert result.exit_code == 0, result.stdout


def test_s():
    result = runner.invoke(
        app,
        [
            "s",
            _TEST_IMAGE_PATH.as_posix(),
            "--transformer",
            'FisheyeFormatEncoder("equidistant") * '
            "Euclidean3DRotator(from_rotation_vector([np.pi / 4, 0, 0])) * "
            'FisheyeFormatDecoder("equidistant")',
            "--radius",
            "max",
            "--out-path",
            (_TEST_DIR / "test.cli.s.jpg").as_posix(),
        ],
    )
    assert result.exit_code == 0, result.stdout

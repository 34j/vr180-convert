from typer.testing import CliRunner

from vr180_convert.cli import app

runner = CliRunner()


def test_help():
    """The help message includes the CLI name."""
    result = runner.invoke(app, ["--help"], prog_name="vr180-convert")
    assert result.exit_code == 0
    assert "vr180-convert" in result.stdout

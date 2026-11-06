import sys
from io import StringIO

from vr180_convert.cli import app


def test_help():
    """The help message includes the CLI name."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        app.help_print()
        output = sys.stdout.getvalue()
        assert "lr" in output or "remap" in output.lower()
    finally:
        sys.stdout = old_stdout

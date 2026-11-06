"""Make the CLI runnable using ``python -m vr180_convert``."""

import sys

from .cli import app

app(sys.argv[1:])

import typer
from rich import print

from .remapper import apply

app = typer.Typer()


@app.command()
def main(from_path: str, to_path: str):
    pass
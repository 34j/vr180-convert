from enum import Enum, IntEnum, StrEnum, auto
import typer
from rich import print

from .remapper import apply, apply_lr
from .transformer import TransformerBase
from .transformer import * # noqa
from typing_extensions import Annotated
app = typer.Typer()

class InterpolationFlags(StrEnum):
    INTER_NEAREST    = auto()
    INTER_LINEAR       = auto()
    INTER_CUBIC        = auto()
    INTER_AREA         = auto()
    INTER_LANCZOS4     = auto()
    INTER_MAX          = auto()
    WARP_FILL_OUTLIERS = auto()
    WARP_INVERSE_MAP   = auto()
    
class BorderTypes(StrEnum):
    BORDER_CONSTANT    = auto()
    BORDER_REPLICATE   = auto()
    BORDER_REFLECT     = auto()
    BORDER_WRAP        = auto()
    BORDER_REFLECT_101 = auto()
    BORDER_TRANSPARENT = auto()
    BORDER_ISOLATED    = auto()

@app.command()
def lr(left_path: Annotated[Path, typer.Argument(help="Left image path")]
       , right_path: Annotated[Path, typer.Argument(help="Right image path")]
       ,transformer: Annotated[str, typer.Option(help="Transformer Python code (to be `eval()`ed)")] = "",
       out_path: Annotated[Path, typer.Option(help="Output image path, defaults to left_path.with_suffix('.out.jpg')")] = Path(""),
    size: Annotated[str, typer.Option(help="Output image size, defaults to 2048x2048")] = "2048x2048",
    interpolation: Annotated[InterpolationFlags, typer.Option(help="Interpolation method, defaults to lanczos4")] = InterpolationFlags.INTER_LANCZOS4,
    boarder_mode: Annotated[BorderTypes, typer.Option(help="Border mode, defaults to constant")] = BorderTypes.BORDER_CONSTANT,
    boarder_value: int = 0,
       
       radius: Annotated[str, typer.Option(help="Radius of the fisheye image, defaults to 'auto'")] = "auto"):
    if transformer == "":
        transformer_ = EquirectangularFormatEncoder() * FisheyeFormatDecoder("equidistant")
    else:
        transformer_ = eval(transformer)
    apply_lr(
        transformer=transformer_,
        left_path=left_path,
        right_path=right_path,
        out_path=Path(left_path).with_suffix(".out.jpg") if out_path == Path("") else out_path,
        radius=float(radius) if radius not in ["auto", "max"] else radius,
        size=tuple(map(int, size.split("x"))),
        interpolation=getattr(cv, interpolation.upper()),
        boarder_mode=getattr(cv, boarder_mode.upper()),
        boarder_value=boarder_value
    )


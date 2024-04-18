from vr180_convert.main import FisheyeFormatDecoder, FisheyeFormatEncoder, apply


def test_main():
    apply(
        "test.jpg",
        "test.out.jpg",
        FisheyeFormatEncoder("rectilinear") * FisheyeFormatDecoder("equidistant"),
    )

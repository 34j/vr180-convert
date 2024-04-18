from vr180_convert.main import FisheyeFormatDecoder, ZoomTransformer, apply, NormalizeTransformer, DenormalizeTransformer, FisheyeFormatEncoder


def test_main():
    apply("test.jpg", "test.out.jpg", FisheyeFormatDecoder("equidistant") *  FisheyeFormatEncoder("rectilinear"))

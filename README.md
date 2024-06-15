# VR180 image converter

<p align="center">
  <a href="https://github.com/34j/vr180-convert/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/vr180-convert/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://vr180-convert.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/vr180-convert.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/vr180-convert">
    <img src="https://img.shields.io/codecov/c/github/34j/vr180-convert.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/vr180-convert/">
    <img src="https://img.shields.io/pypi/v/vr180-convert.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/vr180-convert.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/vr180-convert.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://vr180-convert.readthedocs.io" target="_blank">https://vr180-convert.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/vr180-convert" target="_blank">https://github.com/34j/vr180-convert </a>

---

Simple VR180 image converter on top of OpenCV and NumPy.

## Installation

Install this via pip (or your favourite package manager):

```shell
pipx install vr180-convert
```

## Usage

Simply run the following command to convert 2 fisheye images to a SBS equirectangular VR180 image:

```shell
v1c lr left.jpg right.jpg
```

| left.jpg                       | right.jpg                       | Output                                               |
| ------------------------------ | ------------------------------- | ---------------------------------------------------- |
| ![left](docs/_static/test.jpg) | ![right](docs/_static/test.jpg) | ![output](docs/_static/test.lr.PolynomialScaler.jpg) |

If left and right image paths are the same, the image is divided into two halves (left and right, SBS) and processed as if they were separate images.

## Advanced usage

### Automatic image search

If one of left or right image path is a directory, the program will search for the closest image (in terms of creation time) in the other directory.

```shell
v1c lr left.jpg right_dir
v1c lr left_dir right.jpg
```

Since clocks on cameras may not be very accurate in some cases, it is recommended to check how quickly the clocks of the two cameras shift, and synchronize the clocks before shooting.
However, it can be adjusted by specifying `-ac` option.

```shell
v1c lr left.jpg right_dir -ac 1 # the clock of the right camera is 1 second faster / ahead
v1c lr left_dir right.jpg -ac 1 # the clock of the right camera is 1 second faster / ahead
```

### Radius estimation

The radius of the non-black area of the input image is assumed by counting black pixels by default, but it would be better to specify it manually to get stable results:

```shell
v1c lr left.jpg right.jpg --radius 1000
v1c lr left.jpg right.jpg --radius max # min(width, height) / 2
```

### Calibration

[Rotation matching using the least-squares method](https://lisyarus.github.io/blog/posts/3d-shape-matching-with-quaternions.html) can be performed by clicking corresponding points that can be regarded as infinitely far away from the camera.

```shell
v1c lr left.jpg right.jpg --automatch gui
```

You can also specify the corresponding points manually:

```shell
v1c lr left.jpg right.jpg --automatch "0,0;0,0;1,1;1,1" # left_x1,left_y1;right_x1,right_y1;...
```

$$
a_k, b_k \in \mathbb{R}^3,
\min_{R \in SO(3)} \sum_k \|R a_k - b_k\|^2
$$

<details>
  <summary>Derivation</summary>

#### Derivation

It is known that

$$
\exists_2 q \in SU(2) \subset \mathbb{R}^4, R \in SO(3) \subset \mathbb{R}^{3 \times 3}, \forall a \in \mathbb{R}^3, R a = q * (a, 0) * q^{-1}
$$

where

$$
\forall (v_0, v), (w_0, w) \in \mathbb{R}^4, (v_0, v) * (w_0, w) := (v_0 w_0 - v \cdot w, v_0 w + w_0 v + v \times w)
$$

which shows that

$$
f_l, f_r: SU(2) \to SO(3) \text{ or } \mathbb{R}^{3} \to \mathbb{R}^{3 \times 3}, (q_0, q_1, q_2, q_3) \mapsto \begin{pmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & -q_3 & q_2 \\ q_2 & q_3 & q_0 & -q_1 \\ q_3 & -q_2 & q_1 & q_0 \end{pmatrix}, \begin{pmatrix} q_0 & -q_1 & -q_2 & -q_3 \\ q_1 & q_0 & q_3 & -q_2 \\ q_2 & -q_3 & q_0 & q_1 \\ q_3 & q_2 & -q_1 & q_0 \end{pmatrix} \\
\forall p, q \in SU(2) \text{ or } \mathbb{R}^{3}, p * q = f_l(p) \begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \end{pmatrix} = f_r(q) \begin{pmatrix} p_0 \\ p_1 \\ p_2 \\ p_3 \end{pmatrix}
$$

Using this

$$
\begin{align}
E(q) &:= \sum_k \|R a_k - b_k\|^2 \\
&= \sum_k \|q * a_k * q^{-1} - b_k\|^2 \\
&= \sum_k \|(q * a_k - b_k * q) * q^{-1}\|^2 \\
&= \sum_k \|q * a_k - b_k * q\|^2 \| q^{-1} \|^2 \\
&= \sum_k \|q * a_k - b_k * q\|^2 \\
&= \sum_k \|f_r(a_k) q - f_l(b_k) q\|^2 \\
&= \sum_k \|(f_r(a_k) - f_l(b_k)) q\|^2 \\
&= \sum_k q^T (f_r(a_k) - f_l(b_k))^T (f_r(a_k) - f_l(b_k)) q \\
&= q^T \left( \sum_k (f_r(a_k) - f_l(b_k))^T (f_r(a_k) - f_l(b_k)) \right) q \\
&= q^T B q
\end{align}
$$

($E(q)$ is a quadratic form.)

As $B$ is symmetric, there exists an orthogonal matrix $P$ such that $B = P^T \Lambda P$ where $\Lambda = \mathrm{diag} \{\lambda_i\} (\lambda_1 < \dots < \lambda_4)$ is diagonal. Let $r = P q$, then

$$
E(q) = (P q)^T D (P q) = r^T D r = \sum_i \lambda_i r_i^2
$$

under

$$
\| r \| = r^T r = q^T P^T P q = q^T q = 1
$$

and $E(q)$ is minimized when $r = (1, 0, 0, 0)^T$ and $q = P (1, 0, 0, 0)^T = v_1$ where $v_1$ is the eigenvector corresponding to $\lambda_1$.

</details>

### Anaglyph

`--merge` option (which exports as [anaglyph](https://en.wikipedia.org/wiki/Anaglyph_3D) image) can be used to check if the calibration is successful by checking if the infinitely far points are overlapped.

```shell
v1c lr left.jpg right.jpg --automatch gui --merge
```

### Swap

If the camera is mounted upside down, you can simply use the `--swap` option without changing the transformer or other parameters:

```shell
v1c lr left.jpg right.jpg --swap
```

Or the image can be simply swapped using the `swap` command:

```shell
v1c swap rl.jpg
```

in case one notices that the left and right images are swapped after the conversion.

### Convert to Google's format (Photo Sphere XMP Metadata)

This format is special in that it base64-encodes the right-eye image into the metadata of the left-eye image.
Required for Google Photos, etc.

You can convert the image to this format by:

```shell
v1c xmp lr.jpg
```

The [python-xmp-toolkit](https://github.com/python-xmp-toolkit/python-xmp-toolkit) used in this command requires [exempi](https://libopenraw.freedesktop.org/exempi/) to be installed. Note that if this command is called on Windows, it will attempt to install this library and its dependencies and then run the command on WSL using `subprocess`.

#### References

- [imrivera/google\-photos\-vr180\-test: Test for XMP metadata parsing for VR180 pictures in Google Photos](https://github.com/imrivera/google-photos-vr180-test)
- [temoki/make_vr180photo_py: 左眼カメラ画像と右眼カメラ画像を結合して VR180 3D フォトを作成する Python スクリプト](https://github.com/temoki/make_vr180photo_py)

### Custom conversion model

You can also specify the conversion model by adding Python code directly to the `--transformer` option:

```shell
v1c lr left.jpg right.jpg --transformer 'EquirectangularEncoder() * Euclidean3DRotator(from_rotation_vector([0, np.pi / 4, 0])) * FisheyeDecoder("equidistant")'
```

If tuple, the first transformer is applied to the left image and the second transformer is applied to the right image. If a single transformer is given, it is applied to both images.

Please refer to the [API documentation](https://vr180-convert.readthedocs.io/) for the available transformers and their parameters.
For `from_rotation_vector`, please refer to the [numpy-quaternion documentation](https://quaternion.readthedocs.io/en/latest/Package%20API%3A/quaternion/#from_rotation_vector).

### Single image conversion

To convert a single image, use `v1c s` instead.

### Help

For more information, please refer to the help or API documentation:

```shell
v1c --help
```

## Usage as a library

For more complex transformations, it is recommended to create your own `Transformer`.

Note that the transformation is applied in inverse order (new[(x, y)] = old[transform(x, y)], e.g. to decode [orthographic](https://en.wikipedia.org/wiki/Fisheye_lens#Mapping_function) fisheye images, `transform_polar` should be `arcsin(theta)`, not `sin(theta)`.)

```python
from vr180_convert import PolarRollTransformer, apply_lr

class MyTransformer(PolarRollTransformer):
    def transform_polar(
        self, theta: NDArray, roll: NDArray, **kwargs: Any
    ) -> tuple[NDArray, NDArray]:
        return theta**0.98 + theta**1.01, roll

transformer = EquirectangularEncoder() * MyTransformer() * FisheyeDecoder("equidistant")
apply_lr(transformer, left_path="left.jpg", right_path="right.jpg", out_path="output.jpg")
```

## Tips

### How to determine which image is left or right

<!--
- In the left image, the subject faces more to the right.
- In the right image, the subject faces more to the left.
- In other words, in a SBS image, the subject is oriented toward the center.

In anaglyph images,

- The left eye is covered with a red film, so the portion for the left eye is shown in blue.
- The right eye is covered with a blue film, so the portion for the right eye is shown in red.
| Film Color          | <span style="color:red">Red</span>   | <span style="color:blue">Blue</span> |
| Anaglyph Color      | <span style="color:blue">Blue</span> | <span style="color:red">Red</span>   |
-->

|                     | Left                        | Right                       |
| ------------------- | --------------------------- | --------------------------- |
| Subject Orientation | Right                       | Left                        |
| Film Color          | ${\color{red}\text{Red}}$   | ${\color{blue}\text{Blue}}$ |
| Anaglyph Color      | ${\color{blue}\text{Blue}}$ | ${\color{red}\text{Red}}$   |

- In a SBS image, the subject is oriented toward the center.

### How to edit images

This program cannot read RAW files. To deal with white-outs, etc., it is required to process each image with a program such as Photoshop, Lightroom, [RawTherapee](https://rawtherapee.com/downloads/), [Darktable](https://www.darktable.org/install/), etc.

However, this is so exhaustive, so it is recommended to take the images with JPEG format with care to **avoid overexposure** and to **match the settings** of the two cameras, then convert them with this program and edit the converted images.

<details>
<summary>Example of editing in RawTherapee (Light editing)</summary>

1. Rank the left images in RawTherapee.
2. Use this program to convert the images.
3. Edit the converted images in RawTherapee.

</details>

<details>
<summary>Example of editing in Photoshop (Exquisite editing)</summary>

1. Rank the left images in RawTherapee or Lightroom.
2. Open left image as a Smart Object `LRaw`.
3. Add right image as a Smart Object `RRaw`.
4. Make **minimal** corrections just to match the exposure using `Camera Raw Filter`.
5. Make each Smart Object into Smart Objects (`L`, `R`) again and do any image-dependent processing, such as removing the background.
6. Make both images into a single Smart Object (`P`) and process them as a whole.
7. Export as a PNG file.
8. Hide the other Smart Object (`L` or `R`) (created in step 3) in the Smart Object `P` (created in step 4) and save the Smart Object `P`, then export as a PNG file.
9. Use this program to convert the images.

</details>

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.

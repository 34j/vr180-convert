from typing import Sequence

import attrs
import cv2 as cv
import numpy as np
from ivy import Array


@attrs.frozen(kw_only=True)
class MatchResult:
    points1: Sequence[tuple[float, float]]
    """The points in image 1."""
    points2: Sequence[tuple[float, float]]
    """The points in image 2."""
    kp1: Sequence[cv.KeyPoint]
    """The keypoints in image 1."""
    kp2: Sequence[cv.KeyPoint]
    """The keypoints in image 2."""
    matches: Sequence[cv.DMatch]
    """The matches between the keypoints in image 1 and image 2."""


def feature_match_points(
    image1: Array,
    image2: Array,
    *,
    scale: float | None = None,
    feature_detector: cv.Feature2D | None = None,
    matcher: cv.DescriptorMatcher | None = None
) -> MatchResult:
    """
    Match the points in two images.

    Parameters
    ----------
    image1 : Array
        Image 1 of shape (height, width, channels).
    image2 : Array
        Image 2 of shape (height, width, channels).
    scale : float, optional
        Rescale the image to pass to the feature detector
        for faster processing. The default is None.
    feature_detector : cv.Feature2D, optional
        Feature detector to use. The default is cv.AKAZE.create().
    matcher : cv.DescriptorMatcher, optional
        Matcher to use. The default is cv.BFMatcher().

    Returns
    -------
    MatchResult
        The match result containing the points, keypoints and matches.

    """
    feature_detector = feature_detector or cv.AKAZE.create()
    matcher = matcher or cv.BFMatcher()

    # Resize the image if scale is not None
    if scale is not None:
        image1 = cv.resize(
            image1, (int(image1.shape[1] * scale), int(image1.shape[0] * scale))
        )
        image2 = cv.resize(
            image2, (int(image2.shape[1] * scale), int(image2.shape[0] * scale))
        )

    # Detect and compute the keypoints and descriptors
    kp1, des1 = feature_detector.detectAndCompute(image1, None)
    kp2, des2 = feature_detector.detectAndCompute(image2, None)

    # Match the descriptors
    matches = matcher.match(des1, des2)

    # Compute the points from the matches
    points1, points2 = [], []
    for m in matches:
        points1.append(kp1[m.queryIdx].pt)
        points2.append(kp2[m.trainIdx].pt)

    # Scale the points back to the original image size
    points1_ = np.asarray(points1)
    points2_ = np.asarray(points2)
    if scale is not None:
        points1_ /= scale
        points2_ /= scale

    # Return the match result
    return MatchResult(
        points1=points1_,
        points2=points2_,
        kp1=np.asarray(kp1),
        kp2=np.asarray(kp2),
        matches=np.asarray(matches),
    )

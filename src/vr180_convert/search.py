from pathlib import Path


def find_time_matched_image(
    base_image_path: Path, search_path: Path, /, *, search_path_earlier_diff: float = 0
) -> Path:
    """
    Find the time-matched image to the base image in the search path.

    Parameters
    ----------
    base_image_path : Path
        The base image path.
    search_path : Path
        The search path.
    search_path_earlier_diff : float, optional
        The difference in seconds to search for earlier images, by default 0

    Returns
    -------
    Path
        The path of the time-matched image.

    Raises
    ------
    ValueError
        No time-matched image found.

    """
    st_mtime = base_image_path.stat().st_mtime
    left_path_candidates = sorted(
        search_path.rglob(f"*.{base_image_path.suffix}"),
        key=lambda p: abs(p.stat().st_mtime - st_mtime + search_path_earlier_diff),
        reverse=False,
    )
    left_path_candidates = [p for p in left_path_candidates if (p != base_image_path)]
    if len(left_path_candidates) == 0:
        raise ValueError("No time-matched image found.")
    return left_path_candidates[0]

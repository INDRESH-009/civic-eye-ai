'''
downloading Cloudinary images,
fast resizing,
GPS distance,
visual same-location check (ORB + RANSAC),
warping AFTER into BEFORE,
simple change metrics (edges, Laplacian, grayscale).
'''

# app/utils.py
import math
from typing import Tuple, Optional, List

import cv2
import numpy as np
import requests


def fetch_image_bgr(url: str, timeout: float = 10.0) -> np.ndarray:
    """
    Fetch an image from a URL (e.g., Cloudinary) and return a BGR uint8 image (OpenCV format).
    Raises if download or decode fails.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from URL: {url}")
    return img


def resize_long_edge(img: np.ndarray, long_edge: int = 1024) -> np.ndarray:
    """
    Resize image so that max(height, width) == long_edge (keeps aspect).
    Keeps smaller images unchanged.
    """
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    if h >= w:
        new_h = long_edge
        new_w = int(round(w * (long_edge / h)))
    else:
        new_w = long_edge
        new_h = int(round(h * (long_edge / w)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in meters between two lat/lon points."""
    R = 6371000.0  # meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def orb_homography_metrics(
    img_ref_bgr: np.ndarray,
    img_mov_bgr: np.ndarray,
    max_pts: int = 2000,
) -> Tuple[bool, Optional[np.ndarray], float, float, int, int]:
    """
    Compute ORB features & matches, estimate homography (RANSAC).
    Returns:
      ok, H, inlier_ratio, reproj_error, n_inliers, n_matches_kept
    - img_ref_bgr: reference (BEFORE) image
    - img_mov_bgr: moving (AFTER) image (to be warped into reference)
    """
    g1 = cv2.cvtColor(img_ref_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_mov_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_pts, scaleFactor=1.2, nlevels=8)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return False, None, 0.0, float("inf"), 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(d1, d2, k=2)

    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:  # Lowe ratio test
            good.append(m)

    if len(good) < 8:
        return False, None, 0.0, float("inf"), 0, len(good)

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return False, None, 0.0, float("inf"), 0, len(good)

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(1, len(good))

    # Rough reprojection error
    pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1, 3).T  # 3xN
    proj = (H @ pts1_h)
    proj = proj[:2] / proj[2:3]
    err = np.linalg.norm(proj.T - pts2, axis=1).mean() if len(pts2) > 0 else float("inf")

    return True, H, float(inlier_ratio), float(err), inliers, len(good)


def warp_to_reference(img_moving: np.ndarray, H: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """Warp moving image into reference frame using inverse homography."""
    h, w = ref_shape[:2]
    return cv2.warpPerspective(img_moving, np.linalg.inv(H), (w, h), flags=cv2.INTER_LINEAR)


def canny_edge_density(gray: np.ndarray) -> float:
    """Edge pixels per total pixels (simple scalar)."""
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.sum() / 255.0) / float(gray.size)


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of Laplacian (texture/sharpness proxy)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> GRAY."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

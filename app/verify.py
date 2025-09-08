# app/verify.py
"""
Issue-agnostic before/after verifier:
- GPS/visual same-location check (unchanged logic),
- Align AFTER -> BEFORE (use homography only if good, else ECC fallback, else resize),
- CLAHE brightness normalization,
- Crop top 15% (remove banners/sky/overlays),
- Auto-ROI (3x3 grid) selects most "problem-like" region in BEFORE,
- Signals in ROI: SSIM increase, edge drop, std-dev drop, Laplacian drop, background-distance drop,
- Decision: pass any 2 of 5 signals ⇒ resolved,
- Returns reasons + raw metrics + alignment flags for auditability.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from dateutil import parser as dtparser
from skimage.metrics import structural_similarity as ssim

from .utils import (
    fetch_image_bgr,
    resize_long_edge,
    haversine_distance_m,
    orb_homography_metrics,
    warp_to_reference,
    canny_edge_density,
    laplacian_variance,
    grayscale,
)


# ----------------------------- Data models ----------------------------- #
@dataclass
class ImageMeta:
    lat: Optional[float]
    lng: Optional[float]
    timestamp: Optional[str]  # ISO 8601
    image_url: str
    description: Optional[str] = None


@dataclass
class VerifyConfig:
    # Same-location thresholds
    gps_same_loc_m: float = 25.0
    orb_inlier_ratio_min: float = 0.45
    orb_reproj_err_max: float = 10.0  # more forgiving; we also gate by inliers below

    # Legacy center-crop size (unused now, kept for compatibility)
    roi_frac: float = 0.6

    # Resolution thresholds (need any 2 to pass)
    ssim_roi_min: float = 0.75
    edge_drop_min: float = 0.40
    stddev_drop_min: float = 0.30
    laplacian_drop_min: float = 0.30
    bg_dist_drop_min: float = 0.25
    min_signals: int = 2

    # Auto-ROI tiles
    roi_grid_tiles: int = 3


# ----------------------------- Helpers ----------------------------- #
def _parse_iso(ts: Optional[str]):
    if not ts:
        return None
    try:
        return dtparser.isoparse(ts)
    except Exception:
        return None


def _clahe_gray(gray: np.ndarray) -> np.ndarray:
    """Contrast-normalize grayscale for lighting changes."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _ecc_refine(ref_gray: np.ndarray, mov_gray: np.ndarray, warp_init: Optional[np.ndarray] = None):
    """
    Try ECC alignment (affine). Returns 2x3 warp or None if fails.
    Inputs must be uint8 or float32 in [0,1].
    """
    ref = ref_gray.astype(np.float32) / 255.0
    mov = mov_gray.astype(np.float32) / 255.0
    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32) if warp_init is None else warp_init.astype(np.float32)
    try:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
        _, warp = cv2.findTransformECC(ref, mov, warp_matrix, warp_mode, criteria, None, 5)
        return warp
    except cv2.error:
        return None


# ----------------------------- Location check ----------------------------- #
def same_location_check(
    before: ImageMeta, after: ImageMeta, imgB_bgr: np.ndarray, imgA_bgr: np.ndarray, cfg: VerifyConfig
) -> Tuple[bool, float, Dict[str, Any], Optional[np.ndarray]]:
    """
    Returns (same_loc, location_confidence, debug, H)
    - same_loc: bool decision
    - location_confidence: 0..1
    - debug: metrics dictionary
    - H: homography (if visual check succeeded), else None
    """
    debug: Dict[str, Any] = {}
    gps_score = 0.0

    # 1) GPS distance (if provided)
    if (
        before.lat is not None
        and before.lng is not None
        and after.lat is not None
        and after.lng is not None
    ):
        gps_dist = haversine_distance_m(before.lat, before.lng, after.lat, after.lng)
        debug["gps_distance_m"] = gps_dist
        if gps_dist <= cfg.gps_same_loc_m:
            gps_score = 0.9 if gps_dist <= 0.5 * cfg.gps_same_loc_m else 0.75

    # 2) Visual location (ORB -> RANSAC)
    imgB = resize_long_edge(imgB_bgr, 1024)
    imgA = resize_long_edge(imgA_bgr, 1024)

    ok, H, inlier_ratio, reproj_err, n_inliers, n_kept = orb_homography_metrics(imgB, imgA)
    debug.update(
        {
            "orb_inlier_ratio": inlier_ratio,
            "orb_reproj_error": reproj_err,
            "orb_inliers": n_inliers,
            "orb_matches_kept": n_kept,
        }
    )

    visual_ok = (
        (inlier_ratio >= cfg.orb_inlier_ratio_min)
        and (reproj_err <= cfg.orb_reproj_err_max)
        and (n_inliers >= 20)
    )

    # If GPS is strong, allow weaker homography to still be usable
    gps_dist = debug.get("gps_distance_m", None)
    if gps_dist is not None and gps_dist <= cfg.gps_same_loc_m:
        if (inlier_ratio >= 0.25) and (reproj_err <= 10.0) and (n_inliers >= 8):
            visual_ok = True

    visual_score = max(
        0.0, min(1.0, 0.6 * inlier_ratio + 0.4 * (1.0 - min(1.0, reproj_err / max(1.0, cfg.orb_reproj_err_max))))
    )

    # combine
    loc_conf = max(gps_score, visual_score) if gps_score > 0 else visual_score
    same_loc = (gps_score > 0) or visual_ok

    return same_loc, float(loc_conf), debug, H if visual_ok else None


# ----------------------------- Resolution signals ----------------------------- #
def _auto_roi(grayB: np.ndarray, grayA: np.ndarray, tiles: int = 3):
    """Pick the most 'problem-like' tile in BEFORE using edges + Laplacian variance."""
    Hh, Ww = grayB.shape[:2]
    tile_h, tile_w = max(8, Hh // tiles), max(8, Ww // tiles)

    def problem_score(g: np.ndarray) -> float:
        e = canny_edge_density(g)
        l = laplacian_variance(g)
        return 0.6 * e + 0.4 * (l / (l + 1e-6))

    best = None
    best_score = -1e9
    for ty in range(tiles):
        for tx in range(tiles):
            y0, x0 = ty * tile_h, tx * tile_w
            y1, x1 = min(Hh, y0 + tile_h), min(Ww, x0 + tile_w)
            sc = problem_score(grayB[y0:y1, x0:x1])
            if sc > best_score:
                best_score = sc
                best = (y0, y1, x0, x1)

    y0, y1, x0, x1 = best
    cropB = grayB[y0:y1, x0:x1]
    cropA = grayA[y0:y1, x0:x1]
    return cropB, cropA, (y0, y1, x0, x1)


def compute_resolution_signals(
    imgB_bgr: np.ndarray, imgA_warp_bgr: np.ndarray, cfg: VerifyConfig
) -> Dict[str, Any]:
    """
    Issue-agnostic 'fixed?' signals computed in an auto-selected ROI.
    Signals: SSIM increase, edge density drop, std-dev drop, Laplacian drop, background-distance drop.
    """
    # Gray + brightness normalization
    grayB = _clahe_gray(grayscale(imgB_bgr))
    grayA = _clahe_gray(grayscale(imgA_warp_bgr))

    # Remove top band (banners/sky/overlays)
    h, w = grayB.shape
    cut = int(0.15 * h)
    if h - cut >= 16:  # guard against tiny images
        grayB = grayB[cut:, :]
        grayA = grayA[cut:, :]

    # Auto-ROI (3x3 grid by default)
    cropB, cropA, (y0, y1, x0, x1) = _auto_roi(grayB, grayA, tiles=cfg.roi_grid_tiles)
    Hh, Ww = grayB.shape[:2]

    # 1) SSIM over ROI (higher after => more uniform/clean)
    ssim_roi = float(ssim(cropB, cropA))

    # 2) Edge density drop
    edgeB = canny_edge_density(cropB)
    edgeA = canny_edge_density(cropA)
    edge_drop = float((edgeB - edgeA) / max(1e-6, edgeB))

    # 3) Intensity std-dev drop (texture variance)
    stdB, stdA = float(cropB.std()), float(cropA.std())
    std_drop = float((stdB - stdA) / max(1e-6, stdB))

    # 4) Laplacian variance drop (sharp artifacts reduce)
    lapB = laplacian_variance(cropB)
    lapA = laplacian_variance(cropA)
    lap_drop = float((lapB - lapA) / max(1e-6, lapB))

    # 5) Background-distance drop (ROI becomes more similar to surrounding road)
    pad = max(8, min((y1 - y0), (x1 - x0)) // 6)
    y0b = max(0, y0 - pad)
    y1b = min(Hh, y1 + pad)
    x0b = max(0, x0 - pad)
    x1b = min(Ww, x1 + pad)
    ringB = grayB[y0b:y1b, x0b:x1b].copy()
    ringA = grayA[y0b:y1b, x0b:x1b].copy()
    # remove ROI from ring stats (replace with mean)
    ringB[y0:y1, x0:x1] = cropB.mean()
    ringA[y0:y1, x0:x1] = cropA.mean()

    def _tex_dist(a: np.ndarray, b: np.ndarray) -> float:
        return abs(float(a.std()) - float(b.std())) + abs(laplacian_variance(a) - laplacian_variance(b))

    dist_before = _tex_dist(cropB, ringB)
    dist_after = _tex_dist(cropA, ringA)
    bg_dist_drop = (dist_before - dist_after) / max(1e-6, dist_before)

    # Signals & decision
    signals = {
        "ssim_roi_ok": ssim_roi >= cfg.ssim_roi_min,
        "edge_drop_ok": edge_drop >= cfg.edge_drop_min,
        "stddev_drop_ok": std_drop >= cfg.stddev_drop_min,
        "laplacian_drop_ok": lap_drop >= cfg.laplacian_drop_min,
        "bg_dist_drop_ok": bg_dist_drop >= cfg.bg_dist_drop_min,
    }
    n_ok = int(sum(1 for v in signals.values() if v))

    # Confidence (squashed weighted delta above thresholds)
    raw = (
        0.28 * (ssim_roi - cfg.ssim_roi_min)
        + 0.22 * (edge_drop - cfg.edge_drop_min)
        + 0.22 * (std_drop - cfg.stddev_drop_min)
        + 0.18 * (lap_drop - cfg.laplacian_drop_min)
        + 0.10 * (bg_dist_drop - cfg.bg_dist_drop_min)
    )
    resolved_conf = float(1.0 / (1.0 + np.exp(-4.0 * raw)))

    return {
        "roi": {"y0": y0, "y1": y1, "x0": x0, "x1": x1, "cut_top": cut},
        "ssim_roi": ssim_roi,
        "edge_density_before": float(edgeB),
        "edge_density_after": float(edgeA),
        "edge_drop": edge_drop,
        "stddev_before": stdB,
        "stddev_after": stdA,
        "stddev_drop": std_drop,
        "laplacian_before": float(lapB),
        "laplacian_after": float(lapA),
        "laplacian_drop": lap_drop,
        "bg_dist_before": float(dist_before),
        "bg_dist_after": float(dist_after),
        "bg_dist_drop": float(bg_dist_drop),
        "signals": signals,
        "signals_passed": n_ok,
        "resolved_confidence": resolved_conf,
    }


# ----------------------------- Main entry ----------------------------- #
def verify_pair(before: ImageMeta, after: ImageMeta, cfg: Optional[VerifyConfig] = None) -> Dict[str, Any]:
    """
    Main entry: returns dict with same_location, resolved, confidences, reasons, and raw checks.
    """
    cfg = cfg or VerifyConfig()

    # fetch images
    imgB = fetch_image_bgr(before.image_url)
    imgA = fetch_image_bgr(after.image_url)

    # time sanity (won't block decision, just logged)
    tB = _parse_iso(before.timestamp)
    tA = _parse_iso(after.timestamp)
    time_ok = (tB is None or tA is None) or (tA > tB)

    # same location?
    same_loc, loc_conf, loc_dbg, H = same_location_check(before, after, imgB, imgA, cfg)
    checks = {**loc_dbg, "time_order_ok": bool(time_ok)}

    if not same_loc:
        return {
            "same_location": False,
            "location_confidence": loc_conf,
            "resolved": None,
            "resolved_confidence": 0.0,
            "reasons": ["Different location (GPS/visual mismatch)."],
            "checks": checks,
        }

    # -------- Alignment: use H only if it's GOOD; else ECC; else resize --------
    used_homography = False
    used_ecc = False
    use_H = False

    if H is not None:
        # accept homography only if quality is decent
        inlier_ratio = checks.get("orb_inlier_ratio", 0.0)
        reproj_err = checks.get("orb_reproj_error", 1e9)
        if (inlier_ratio >= 0.35) and (reproj_err <= 10.0):
            use_H = True

    if use_H:
        used_homography = True
        imgA_warp = warp_to_reference(imgA, H, imgB.shape)
    else:
        # ECC fallback on brightness-normalized grayscale
        gB = _clahe_gray(grayscale(imgB))
        gA0 = cv2.resize(imgA, (imgB.shape[1], imgB.shape[0]), interpolation=cv2.INTER_LINEAR)
        gA = _clahe_gray(grayscale(gA0))
        warp = _ecc_refine(gB, gA)
        if warp is not None:
            used_ecc = True
            imgA_warp = cv2.warpAffine(gA0, warp, (imgB.shape[1], imgB.shape[0]), flags=cv2.INTER_LINEAR)
        else:
            imgA_warp = gA0  # fallback resize

    # compute signals & decide
    res = compute_resolution_signals(imgB, imgA_warp, cfg)
    resolved = res["signals_passed"] >= cfg.min_signals

    # human-readable reasons
    reasons = []
    if "gps_distance_m" in checks:
        reasons.append(f"GPS distance {checks['gps_distance_m']:.1f}m")
    if "orb_inlier_ratio" in checks:
        reasons.append(f"RANSAC inlier ratio {checks.get('orb_inlier_ratio', 0.0):.2f}")
    reasons.append(f"SSIM(ROI) {res['ssim_roi']:.2f}")
    reasons.append(f"Edge drop {res['edge_drop']:.2f}")
    reasons.append(f"Std-dev drop {res['stddev_drop']:.2f}")
    reasons.append(f"Laplacian drop {res['laplacian_drop']:.2f}")
    reasons.append(f"BG-dist drop {res['bg_dist_drop']:.2f}")

    return {
        "same_location": True,
        "location_confidence": float(loc_conf),
        "resolved": bool(resolved),
        "resolved_confidence": float(res["resolved_confidence"]),
        "reasons": reasons,
        "checks": {**checks, **res, "used_homography": used_homography, "used_ecc": used_ecc},
    }

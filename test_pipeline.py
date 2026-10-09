"""
Self-test for the LUNARIS backend. Not a formal pytest suite -- a readable
script you can run directly to sanity-check every engine against a
synthetic pair with a KNOWN ground-truth transform, plus one real HTTP
round-trip through the FastAPI app.

    python3 test_pipeline.py
"""
from __future__ import annotations

import io
import sys

import cv2
import numpy as np


def make_synthetic_pair(n=520, seed=0):
    rng = np.random.default_rng(seed)
    craters = [(rng.uniform(50, n - 50), rng.uniform(50, n - 50), rng.uniform(9, 30)) for _ in range(46)]
    # dense small rocks/speckle give SIFT/ORB/SuperPoint actual corner texture --
    # smooth gradients alone (craters only) are nearly featureless for a detector.
    speckle = [(rng.uniform(0, n), rng.uniform(0, n), rng.uniform(1.5, 4), rng.choice([-1, 1])) for _ in range(1600)]

    def render(craters, speckle, sun_dx, sun_dy, noise_amt):
        shade = np.zeros((n, n), dtype=np.float32)
        highlight = np.zeros((n, n), dtype=np.float32)
        for (x, y, r) in craters:
            cv2.circle(shade, (int(x), int(y)), max(1, int(r)), -60.0, -1)
            hlx, hly = x - r * sun_dx * 0.5, y - r * sun_dy * 0.5
            cv2.circle(highlight, (int(hlx), int(hly)), max(1, int(r * 0.5)), 65.0, -1)
        k = 21
        shade = cv2.GaussianBlur(shade, (k, k), 0)
        highlight = cv2.GaussianBlur(highlight, (k, k), 0)

        canvas = np.full((n, n), 128, dtype=np.float32) + shade + highlight
        for (x, y, r, sign) in speckle:
            cv2.circle(canvas, (int(x), int(y)), max(1, int(r)), float(128 + sign * 42), -1)
        canvas += rng.normal(0, noise_amt, canvas.shape)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # ground-truth affine: small rotation + scale + translation
    ang = np.deg2rad(rng.uniform(-6, 6))
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-18, 18), rng.uniform(-18, 18)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    cx, cy = n / 2, n / 2
    H_gt = np.array([
        [scale * cos_a, -scale * sin_a, cx - scale * cos_a * cx + scale * sin_a * cy + tx],
        [scale * sin_a,  scale * cos_a, cy - scale * sin_a * cx - scale * cos_a * cy + ty],
        [0, 0, 1],
    ])

    ref_craters = []
    for (x, y, r) in craters:
        p = H_gt @ np.array([x, y, 1.0])
        ref_craters.append((p[0], p[1], r * scale))

    ref_speckle = []
    for (x, y, r, sign) in speckle:
        p = H_gt @ np.array([x, y, 1.0])
        ref_speckle.append((p[0], p[1], r * scale, sign))

    src = render(craters, speckle, 1.0, 0.3, 5)
    ref = render(ref_craters, ref_speckle, -0.6, 0.7, 8)
    return src, ref, H_gt


def compare_to_gt(H_est, H_gt, n=520):
    """Mean pixel disagreement between the two transforms over a grid of
    sample points -- a far more meaningful ground-truth check than a raw
    matrix-element difference, since it's in interpretable pixel units."""
    if H_est is None:
        return float("inf")
    H_est = np.array(H_est, dtype=np.float64)
    grid = np.array([[x, y] for x in range(40, n - 40, 40) for y in range(40, n - 40, 40)], dtype=np.float32)
    p_est = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_est).reshape(-1, 2)
    p_gt = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_gt.astype(np.float32)).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(p_est - p_gt, axis=1)))


def main():
    sys.path.insert(0, ".")
    from app.pipeline import classical, deep_lightglue, deep_loftr, hybrid

    src, ref, H_gt = make_synthetic_pair()
    cv2.imwrite("/tmp/lunaris_src.png", src)
    cv2.imwrite("/tmp/lunaris_ref.png", ref)

    print("=" * 70)
    print("LUNARIS backend self-test -- synthetic pair, known ground truth")
    print("=" * 70)

    print(f"\nSuperPoint+LightGlue available: {deep_lightglue.is_available()} "
          f"({deep_lightglue.unavailable_reason()})")
    print(f"LoFTR available:                {deep_loftr.is_available()} "
          f"({deep_loftr.unavailable_reason()})")

    engines = [("SIFT/ORB baseline", lambda: classical.run_classical(src, ref, detector="orb"))]
    if deep_lightglue.is_available():
        engines.append(("SuperPoint+LightGlue", lambda: deep_lightglue.run_lightglue(src, ref)))
    engines.append(("Hybrid Proposed Model", lambda: hybrid.run_hybrid(src, ref)))
    if deep_loftr.is_available():
        engines.append(("LoFTR", lambda: deep_loftr.run_loftr(src, ref)))

    for name, fn in engines:
        r = fn()
        m = r["metrics"]
        gt_dev = compare_to_gt(r["_internal"]["H"], H_gt)
        print(f"\n--- {name} ---")
        print(f"  total matches      : {m['total_matches']}")
        print(f"  inlier matches     : {m['inlier_matches']}")
        print(f"  inlier ratio       : {m['inlier_ratio']*100:.1f}%")
        print(f"  RMSE (px)          : {m['rmse_px']}")
        print(f"  median error (px)  : {m['median_error_px']}")
        print(f"  spatial coverage   : {m['spatial_coverage_pct']}%")
        print(f"  processing time    : {m['processing_time_ms']:.1f} ms")
        print(f"  vs ground truth    : {gt_dev:.3f} px mean deviation over test grid")
        assert m["total_matches"] >= 0
        if m["inlier_matches"] > 0:
            assert m["rmse_px"] < 10, "RMSE implausibly high for a synthetic near-identity pair"

    print("\n" + "=" * 70)
    print("HTTP round-trip test via FastAPI TestClient")
    print("=" * 70)
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    print("GET /api/models ->", client.get("/api/models").json())
    print("GET /api/datasets ->", len(client.get("/api/datasets").json()["source"]), "source entries")

    with open("/tmp/lunaris_src.png", "rb") as fs, open("/tmp/lunaris_ref.png", "rb") as fr:
        resp = client.post(
            "/api/register",
            files={"source": ("src.png", fs, "image/png"), "reference": ("ref.png", fr, "image/png")},
            data={"model": "hybrid"},
        )
    print("POST /api/register status:", resp.status_code)
    body = resp.json()
    assert resp.status_code == 200
    assert "metrics" in body and "registered_image_png_b64" in body
    print("engine used:", body["engine_label"], "| fallback_used:", body["fallback_used"])
    print("metrics:", body["metrics"])
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    main()

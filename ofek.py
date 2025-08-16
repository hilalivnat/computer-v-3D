#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_volume_tool_documented.py
=============================

Interactive single-view metrology tool to estimate **box volumes** from **one photograph**.
The workflow relies on standard Computer Vision 3D concepts (as in Hartley & Zisserman):
- Planar homography and **rectification** of a selected face (projective geometry).
- A user-provided **metric** on one edge (known physical length) to convert pixels↔centimeters.
- Repeating the process on a **second orthogonal face** to recover the height.
- Robust **point picking** with OpenCV HighGUI (with visual feedback, undo/reset, etc.).

USAGE (quick start)
-------------------
1) Install dependencies: `pip install opencv-python numpy`
2) Run from a terminal:
   `python box_volume_tool_documented.py --image box1-4.jpg`
   (You can also pass `--image box1-5.jpg` or your own path)

3) In the console choose a box: `brown / black / blue`.
4) **Face A** (any visible rectangular face):
   - In the image window: Left-click 4 corners (clockwise starting near top-left).
   - Press **Enter** to accept (Right-click = undo, `r` = reset, `q`/Esc = cancel and reprompt).
   - A rectified face appears → click **two endpoints on a single known edge** → Enter.
   - In console: type the real-world length (cm) for that edge from the offered list and press Enter.
5) **Face B** (a face that **shares an edge** with Face A; preferably the side face):
   - Repeat the same steps (4 corners, then 2 points on a known edge and enter its length).
6) The script prints estimated (L, W, H) in cm and the **volume in cm^3**.
   Optionally save the rectified faces and a JSON summary.

CONTROLS (OpenCV window)
------------------------
- Left click:  add point
- Right click: undo last point
- Enter:       finish the current step (requires the exact number of points)
- r:           reset the current step
- q or Esc:    cancel the current step (the app will reprompt)
- Resize the window with the mouse to zoom in/out for precise clicks.

NOTES
-----
- You **do not** have to pick the top face; any visible rectangular face works.
- For the brown box for example: top has edges 12cm and 10cm, side has 11.5cm (height).
- The tool does **not** assume perfect orthogonality; it finds the shared dimension between
  the two rectified faces by nearest match and treats the other dimension as the height.
- To reduce noise, click a few pixels inside the corners (and not exactly on the very edge).

Author: (you)
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Configuration / presets
# ---------------------------------------------------------------------------

# Optional convenience: when True, the point picker will automatically finish
# as soon as the required number of points has been selected (no need to press Enter).
AUTO_FINISH = False

# Known dimensions for each box (centimeters). These are used for calibration choices
# and for printing a reference volume at the end (for sanity checking).
BOX_DIM_PRESETS = {
    "brown": {  # small Kleenex-like tissue box
        "known_lengths_cm": [10.0, 11.5, 12.0],
        "expected_full_dims_cm": (12.0, 10.0, 11.5),  # (L, W, H) reference only
    },
    "black": {  # Nisko bulbs box
        "known_lengths_cm": [6.3, 11.5, 18.2],
        "expected_full_dims_cm": (18.2, 11.5, 6.3),
    },
    "blue": {   # large Kleenex-like tissue box
        "known_lengths_cm": [9.0, 12.5, 22.3],
        "expected_full_dims_cm": (22.3, 12.5, 9.0),
    }
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeasuredFace:
    """Container for a rectified face and its metric scale."""
    name: str                        # "A" or "B"
    warped: np.ndarray               # rectified (fronto-parallel) face image
    px_size: Tuple[float, float]     # (width_px, height_px) after rectification
    scale_cm_per_px: float           # centimeters per pixel along the rectified axes
    dims_cm: Tuple[float, float]     # (width_cm, height_cm) of the rectified face

@dataclass
class BoxResult:
    """Final result for one measured box (using two faces)."""
    box_name: str
    image_path: str
    faceA: Optional[MeasuredFace]
    faceB: Optional[MeasuredFace]
    estimated_dims_cm: Optional[Tuple[float, float, float]]  # (L, W, H)
    volume_cm3: Optional[float]

# ---------------------------------------------------------------------------
# Geometry helpers (projective geometry / H&Z)
# ---------------------------------------------------------------------------

def order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    """
    Given four 2D points clicked roughly around a quadrilateral, return them
    ordered as [top-left, top-right, bottom-right, bottom-left] **clockwise**.

    Implementation:
    - Compute the centroid and sort by angle to get CCW order.
    - Rotate the list so it starts near the top-left (smallest x+y).
    - If the order is CCW, flip to CW (so homography uses a consistent mapping).
    """
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)  # CCW
    ordered = pts[order]
    idx = np.argmin(ordered[:, 0] + ordered[:, 1])  # start near top-left
    ordered = np.roll(ordered, -idx, axis=0)

    # Ensure clockwise orientation (swap order if needed)
    v1 = ordered[1] - ordered[0]
    v2 = ordered[2] - ordered[1]
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross > 0:  # CCW → convert to CW
        ordered = np.array([ordered[0], ordered[3], ordered[2], ordered[1]], dtype=np.float32)
    return ordered

def four_point_transform(image: np.ndarray, pts: np.ndarray):
    """
    Compute a planar homography that maps a clicked quadrilateral to a
    fronto-parallel rectangle, and warp the image accordingly.

    Steps (classic "document scan" approach):
    - Order the points with `order_points_clockwise`.
    - Compute the target rectangle width/height as the max of opposite edges.
    - Use `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to rectify.
    Returns: (warped_image, H)
    """
    rect = order_points_clockwise(pts)
    (tl, tr, br, bl) = rect

    # Estimate target rectangle size from opposite edge lengths
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth  = int(round(max(widthA, widthB)))
    maxHeight = int(round(max(heightA, heightB)))

    # Destination rectangle in pixel coordinates
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype=np.float32)

    # Homography and warp
    H = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, H, (maxWidth, maxHeight))
    return warped, H

def euclidean(p0, p1) -> float:
    """Simple Euclidean distance between two 2D points."""
    p0 = np.array(p0); p1 = np.array(p1)
    return float(np.linalg.norm(p1 - p0))

# ---------------------------------------------------------------------------
# Interactive picking (OpenCV GUI) with live feedback
# ---------------------------------------------------------------------------

class PointPicker:
    """
    Small interactive helper for collecting an exact number of points from an image,
    with visual feedback (circles + indices + polylines), undo/reset, and finish/cancel.

    Controls:
    - Left click: add point
    - Right click: undo last
    - Enter: finish when count matches `need`
    - r: reset
    - q/Esc: cancel (the caller will reprompt)
    """
    def __init__(self, img_bgr: np.ndarray, window: str, need: int, connect=True):
        self.orig = img_bgr.copy()  # immutable background
        self.img  = img_bgr.copy()  # working buffer with overlays
        self.win = window
        self.need = need            # required number of points
        self.connect = connect      # whether to draw connecting lines
        self.pts: List[Tuple[int, int]] = []
        self.finished = False
        self.cancelled = False

        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        self._redraw()
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _overlay_text(self):
        """Render instructions at the top of the window."""
        msg = f"Pick {self.need} points (L-click add, R-click undo, 'r' reset, Enter finish, 'q'/Esc cancel)"
        cv2.putText(self.img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,20,20), 3, cv2.LINE_AA)
        cv2.putText(self.img, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)

    def _draw_pts(self):
        """Draw current points and (optionally) connecting segments."""
        for i, (x, y) in enumerate(self.pts):
            cv2.circle(self.img, (int(x), int(y)), 6, (0, 0, 255), -1)
            cv2.putText(self.img, str(i+1), (int(x)+8, int(y)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(self.img, str(i+1), (int(x)+8, int(y)-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2, cv2.LINE_AA)
        if self.connect and len(self.pts) >= 2:
            for i in range(len(self.pts) - 1):
                cv2.line(self.img, self.pts[i], self.pts[i+1], (0, 0, 255), 2)

    def _redraw(self):
        """Refresh the window with current overlays."""
        self.img = self.orig.copy()
        self._overlay_text()
        self._draw_pts()
        cv2.imshow(self.win, self.img)

    def _on_mouse(self, event, x, y, flags, param):
        """Mouse callback: add/undo points with immediate visual feedback."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((x, y))
            if len(self.pts) > self.need:
                self.pts = self.pts[:self.need]
            self._redraw()
            if AUTO_FINISH and len(self.pts) == self.need:
                self.finished = True
                cv2.destroyWindow(self.win)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.pts:
                self.pts.pop()
                self._redraw()

    def run(self) -> List[Tuple[float, float]]:
        """
        Event loop: wait for Enter (finish), r (reset), or q/Esc (cancel).
        Returns the collected points; an empty list denotes cancellation.
        """
        while True:
            key = cv2.waitKey(20) & 0xFFFF
            if self.finished:
                break
            if key in (13, 10):                # Enter
                if len(self.pts) == self.need:
                    self.finished = True
                    break
            elif key in (27, ord('q'), ord('Q')):  # Esc or q
                self.cancelled = True
                break
            elif key in (ord('r'), ord('R')):  # reset
                self.pts.clear()
                self._redraw()
        cv2.destroyWindow(self.win)
        if self.cancelled:
            return []
        return [(float(x), float(y)) for (x, y) in self.pts]

def pick_points_cv(img_bgr: np.ndarray, need: int, title: str, connect=True) -> List[Tuple[float, float]]:
    """
    Helper wrapper around PointPicker that reprompts automatically if the user cancels
    or confirms with an incorrect number of points.
    """
    picker = PointPicker(img_bgr, title, need, connect=connect)
    while True:
        pts = picker.run()
        if len(pts) == need:
            return pts
        picker = PointPicker(img_bgr, title, need, connect=connect)

# ---------------------------------------------------------------------------
# Measurement workflow
# ---------------------------------------------------------------------------

def pick_face_and_calibrate(image_bgr: np.ndarray, face_name: str,
                            known_lengths_cm: List[float]) -> MeasuredFace:
    """
    Pick and rectify one **planar face** of the box, then set a **metric scale**
    by clicking two endpoints of a **known physical edge** on the rectified face.

    Returns a MeasuredFace with:
      - the rectified image,
      - centimeters-per-pixel scale,
      - the approximate face dimensions in centimeters.
    """
    # 1) User picks 4 corners of a visible rectangular face in the original image
    pts = pick_points_cv(
        image_bgr, 4,
        f"Face {face_name}: pick 4 VISIBLE corners (clockwise from near top-left)",
        connect=True
    )

    # 2) Rectify that face via homography → fronto-parallel rectangle
    warped, _ = four_point_transform(image_bgr, np.array(pts, dtype=np.float32))
    h, w = warped.shape[:2]

    # 3) Calibrate pixels→centimeters using a known edge on the rectified face
    p2 = pick_points_cv(
        warped, 2,
        f"Face {face_name} rectified: pick two endpoints of a KNOWN edge",
        connect=False
    )
    px_len = euclidean(p2[0], p2[1])
    print(f"[Face {face_name}] known edge length in pixels: {px_len:.2f}px")

    # 4) Ask the user to provide the real-world length for the clicked edge
    print(f"[Face {face_name}] choose real length from: {known_lengths_cm}")
    while True:
        try:
            real_cm = float(input(f"Enter one of {known_lengths_cm}: ").strip())
            if real_cm in known_lengths_cm:
                break
            print("Please enter an exact value from the list.")
        except Exception:
            print("Invalid input, try again.")

    # 5) Compute centimeters-per-pixel and convert the rectified rectangle size
    cm_per_px = real_cm / px_len
    dim_x = w * cm_per_px
    dim_y = h * cm_per_px
    print(f"[Face {face_name}] scale = {cm_per_px:.5f} cm/px → rectified size ≈ ({dim_x:.2f} × {dim_y:.2f}) cm")

    return MeasuredFace(face_name, warped, (float(w), float(h)), cm_per_px, (dim_x, dim_y))

def estimate_box_volume_from_faces(faceA: MeasuredFace, faceB: MeasuredFace):
    """
    Given two measured faces (A and B), infer (L, W, H) and the volume.
    We assume exactly one dimension is **shared** between face A and face B.
    Strategy: sort each face dims (descending), try all 4 pairings, pick the
    pairing with the smallest absolute difference for the shared dimension.

    Returns: (dims, volume, mismatch_error_cm)
             dims = (L, W, H) with L≥W taken from faceA's rectified axes.
             volume in cm^3.
             mismatch_error_cm is a diagnostic (0 is perfect, larger means noisier clicks).
    """
    # Sort dims so the first entry is the larger one
    a, b = sorted(faceA.dims_cm, reverse=True)  # a ≥ b
    c, d = sorted(faceB.dims_cm, reverse=True)  # c ≥ d

    # Try all hypotheses on which dimension is shared
    candidates = [
        (a, abs(a - c), d),  # assume 'a' (A) matches 'c' (B) → height = d
        (a, abs(a - d), c),  # assume 'a' matches 'd' → height = c
        (b, abs(b - c), d),  # assume 'b' matches 'c' → height = d
        (b, abs(b - d), c),  # assume 'b' matches 'd' → height = c
    ]

    # Select the hypothesis with the smallest discrepancy
    shared, err, H = min(candidates, key=lambda t: t[1])

    # Define L, W from faceA (so L ≥ W by construction)
    L, W = a, b
    dims = (L, W, H)
    vol = L * W * H
    return dims, vol, err

# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------

def main(image_path: str):
    """Load the image, run the interactive workflow, and save/report results."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    print("Loading:", image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("OpenCV failed to read the image.")

    results: List[BoxResult] = []
    print("\nBoxes: brown / black / blue   (type 'done' to finish)\n")
    while True:
        name = input("Box name (brown/black/blue or done): ").strip().lower()
        if name == "done":
            break
        if name not in BOX_DIM_PRESETS:
            print("Unknown box.")
            continue

        known = BOX_DIM_PRESETS[name]["known_lengths_cm"]
        print(f"\n--- {name.upper()} ---")
        print("Face A: pick ANY visible rectangular face of this box.")
        faceA = pick_face_and_calibrate(img, "A", known)

        print("\nFace B: pick a face that shares an edge with Face A (ideally the side)." )
        faceB = pick_face_and_calibrate(img, "B", known)

        # Estimate dimensions and volume from the two faces
        dims, vol, err = estimate_box_volume_from_faces(faceA, faceB)
        print(f"\n[{name}] Estimated (L,W,H) ≈ ({dims[0]:.2f}, {dims[1]:.2f}, {dims[2]:.2f}) cm")
        print(f"[{name}] Estimated volume ≈ {vol:.2f} cm^3  (shared-edge mismatch ≈ {err:.2f} cm)")

        # Print reference volume from the preset (for sanity checking only)
        exp = BOX_DIM_PRESETS[name].get("expected_full_dims_cm")
        if exp:
            v_exp = exp[0] * exp[1] * exp[2]
            print(f"[{name}] Reference dims from prompt: {exp} → {v_exp:.2f} cm^3")

        results.append(BoxResult(name, image_path, faceA, faceB, dims, vol))

        # Save rectified face images?
        save = input("Save rectified faces? (y/n): ").strip().lower()
        if save == "y":
            out_dir = os.path.join(os.path.dirname(image_path) or ".", "rectified_faces")
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, f"{name}_A.png"), faceA.warped)
            cv2.imwrite(os.path.join(out_dir, f"{name}_B.png"), faceB.warped)
            print("Saved to:", out_dir)

    # Persist a JSON summary of all runs
    if results:
        out_json = os.path.join(os.path.dirname(image_path) or ".", "box_volume_results.json")
        payload = []
        for r in results:
            payload.append({
                "box": r.box_name,
                "image": r.image_path,
                "faceA_dims_cm": [round(x, 3) for x in r.faceA.dims_cm],
                "faceB_dims_cm": [round(x, 3) for x in r.faceB.dims_cm],
                "estimated_dims_cm": [round(x, 3) for x in r.estimated_dims_cm],
                "volume_cm3": round(r.volume_cm3, 3),
            })
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print("\nWrote summary:", out_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Box volume from single image via homography rectification.")
    parser.add_argument("--image", "-i", type=str, default="box1-4.jpg",
                        help="Path to image (e.g., box1-4.jpg or box1-5.jpg)")
    args = parser.parse_args()
    main(args.image)

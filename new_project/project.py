import cv2
import numpy as np
from glob import glob
from pathlib import Path

# ---------- User parameters ----------
CHESSBOARD = (7, 5)         # inner corners (cols, rows)
SQUARE_SIZE = 25.0          # mm
IMG_DIR = "single"
IMG_PATTERN = "*.jpg"
OUT_NPZ = "single_calib.npz"
# ------------------------------------

# ---- Single-camera calibration (intrinsics K, distortion) ----
objp = np.zeros((CHESSBOARD[1] * CHESSBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints = [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
flags = cv2.CALIB_RATIONAL_MODEL

images = sorted(glob(str(Path(IMG_DIR) / IMG_PATTERN)))
if not images:
    raise RuntimeError(f"No images found in: {IMG_DIR}/{IMG_PATTERN}")

img_size = None
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"[WARN] Could not read {fname}, skipping")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_size is None:
        img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if ret:
        corners_refined = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
        )
        imgpoints.append(corners_refined)
        objpoints.append(objp)
    else:
        print(f"[INFO] Chessboard NOT found: {fname}")

print(f"Detected chessboard in {len(imgpoints)} / {len(images)} images")
if len(imgpoints) < 5:
    raise RuntimeError("Not enough valid detections. Capture more views.")

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None, flags=flags
)
print("\n=== Single Camera Calibration Results ===")
print("RMS reprojection error:", ret)
print("K (camera matrix):\n", K)
print("dist (distortion coefficients):\n", dist.ravel())

# ====================== USER INPUTS ======================
# Pixel correspondences of the box corners (same order in both images):
pts1 = np.array([
    (1068, 2155),
    (1235, 2993),
    (1991, 2496),
    (2092, 1650),
    (1436, 1210),
    (537, 1555)
], dtype=np.float32)

pts2 = np.array([
    (579, 1545),
    (757, 2390),
    (1501, 2906),
    (1571, 2123),
    (2169, 1475),
    (1298, 1113)
], dtype=np.float32)

# ---- NEW: known baseline between the two camera centers ----
BASELINE_MM = 420.0  # ~42 cm
EDGE_TRIPLET = (0, 1, 3, 4)  # for volume (three edges from a common corner)
# ============================================================

def estimate_pose_from_corresp(K, p1, p2):
    """RANSAC E → recover R,t (t is unit-length)."""
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("findEssentialMat failed; check correspondences/K.")
    in1 = p1[mask.ravel() == 1]
    in2 = p2[mask.ravel() == 1]
    _, R, t, _ = cv2.recoverPose(E, in1, in2, K)
    return R, t  # ||t|| = 1

def triangulate_with_baseline(K, R, t_unit, p1, p2, baseline_mm):
    """
    Build P1, P2 with translation scaled to the known baseline (in mm),
    then triangulate. Returns 3D points in **mm**.
    """
    # Projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    t_scaled = t_unit * baseline_mm
    P2 = K @ np.hstack([R, t_scaled])

    # Triangulate (expects 2xN)
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)
    X = (pts4d[:3] / pts4d[3]).T  # in millimeters, thanks to scaled baseline
    return X

def volume_from_edges(X, triplet):
    i0, iX, iY, iZ = triplet
    v1 = X[iX] - X[i0]
    v2 = X[iY] - X[i0]
    v3 = X[iZ] - X[i0]
    vol = abs(np.dot(v1, np.cross(v2, v3)))  # mm^3
    Lx, Ly, Lz = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    return vol, (Lx, Ly, Lz)

# ---------- Pipeline ----------
R, t_unit = estimate_pose_from_corresp(K, pts1, pts2)
X_mm = triangulate_with_baseline(K, R, t_unit, pts1, pts2, BASELINE_MM)

# Sanity: orthogonality check (optional)
i0, iX, iY, iZ = EDGE_TRIPLET
v1 = X_mm[iX] - X_mm[i0]
v2 = X_mm[iY] - X_mm[i0]
v3 = X_mm[iZ] - X_mm[i0]
# for a, b, name in [(v1, v2, "v1·v2"), (v1, v3, "v1·v3"), (v2, v3, "v2·v3")]:
#     cosang = abs(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
#     print(name, "cos≈", float(cosang))

# Volume & edge lengths (now in metric units)
vol_mm3, (Lx, Ly, Lz) = volume_from_edges(X_mm, EDGE_TRIPLET)
print("Edge lengths (mm):", f"{Lx:.2f}", f"{Ly:.2f}", f"{Lz:.2f}")
print("Volume (cm^3):", vol_mm3 / 1000.0)

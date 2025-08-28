import cv2
import numpy as np
from glob import glob
from pathlib import Path

# ---------- User parameters ----------
# Chessboard inner corners: (columns, rows) as OpenCV expects
CHESSBOARD = (7,5)              # e.g., 9x6 inner corners
SQUARE_SIZE = 25.0             # size of one square in mm (or any unit you prefer)
IMG_DIR = "single" # folder with chessboard images for ONE camera
IMG_PATTERN = "*.jpg"          # image filename pattern

# Output
OUT_NPZ = "single_calib.npz"
# ------------------------------------

# Prepare object points for a single chessboard view, z = 0
# Object points will be scaled by SQUARE_SIZE, so units carry through your intrinsics/extrinsics.
objp = np.zeros((CHESSBOARD[1] * CHESSBOARD[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in world space
imgpoints = []  # 2D points in image plane

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
flags = cv2.CALIB_RATIONAL_MODEL  # robust model incl. k3..k6 if needed

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
        img_size = gray.shape[::-1]  # (width, height)

    # Try to find chessboard inner corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHESSBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # Refine corners to subpixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1), criteria=criteria
        )
        imgpoints.append(corners_refined)
        objpoints.append(objp)
    else:
        print(f"[INFO] Chessboard NOT found: {fname}")

print(f"Detected chessboard in {len(imgpoints)} / {len(images)} images")
if len(imgpoints) < 5:
    raise RuntimeError("Not enough valid detections. Capture more views at varied angles/distances.")

# Calibrate
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None, flags=flags
)
print("\n=== Single Camera Calibration Results ===")
print("RMS reprojection error:", ret)
print("K (camera matrix):\n", K)
print("dist (distortion coefficients):\n", dist.ravel())


import cv2
import numpy as np

# ====================== USER INPUTS ======================
CAL_NPZ = "single_calib.npz"   # file with K (and optionally dist)

# Corner pixels in each image (Nx2 each). Order must correspond between images.
# Fill these with YOUR measured pixel coordinates:
pts1 = np.array([
    (1068, 2155),
(1235, 2993),
(1190, 2496),
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

# Choose EXACTLY ONE way to set metric scale:
BASELINE_MM = None        # e.g., 120.0 if the camera moved ~12 cm
EDGE_LEN_MM = 115.0       # e.g., the real length of one top edge of the box (mm)
EDGE_IDX = (0, 1)         # indices (i,j) in pts1/pts2 that correspond to that known edge

# Indices of three edges that meet at one corner (for volume).
# Choose one vertex i0 and its three adjacent vertices iX,iY,iZ
# (these should be corners along the 3 box directions from i0).
EDGE_TRIPLET = (0, 1, 3, 4)
# ========================================================

def load_K(npz_path):
    d = np.load(npz_path)
    return d["K"]

def estimate_pose_from_corresp(K, p1, p2):
    """RANSAC essential matrix -> R,t (t is unit-length, unknown scale)."""
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("findEssentialMat failed; check correspondences/K.")
    in1 = p1[mask.ravel()==1]; in2 = p2[mask.ravel()==1]
    _, R, t, _ = cv2.recoverPose(E, in1, in2, K)
    return R, t  # t has norm 1 (unknown metric scale)

def triangulate_all(K, R, t, p1, p2):
    """Triangulate all correspondences with projection matrices from pose."""
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P1, P2, p1.T, p2.T)  # inputs must be 2xN
    X = (pts4d[:3] / pts4d[3]).T  # Nx3, arbitrary scale if t is unit
    return X

def scale_by_known_edge(X, edge_idx, true_len_mm):
    i, j = edge_idx
    cur_len = np.linalg.norm(X[i] - X[j])
    if cur_len < 1e-9:
        raise RuntimeError("Known edge triangulated with ~zero length; check indices.")
    s = true_len_mm / cur_len
    return X * s

def volume_from_edges(X, triplet):
    """Use triple product of three edges from a common corner (axis-agnostic)."""
    i0, iX, iY, iZ = triplet
    v1 = X[iX] - X[i0]
    v2 = X[iY] - X[i0]
    v3 = X[iZ] - X[i0]
    vol = abs(np.dot(v1, np.cross(v2, v3)))  # mm^3
    # Also return edge lengths for reference
    Lx, Ly, Lz = np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3)
    return vol, (Lx, Ly, Lz)

# ---------- Pipeline ----------
# K = load_K(CAL_NPZ)

def scale_from_multiple_edges(X, known_edges):
    est = []; true = []
    for (i,j,Ltrue) in known_edges:
        Lest = np.linalg.norm(X[i]-X[j])
        if Lest > 1e-6:
            est.append(Lest); true.append(Ltrue)
    est = np.array(est); true = np.array(true)
    # s*est ≈ true  ->  s = (est·true) / (est·est)
    s = (est@true) / (est@est)
    return X * s, s
# 1) Pose up to scale from your 2D–2D correspondences
R, t = estimate_pose_from_corresp(K, pts1, pts2)  # t has ||t|| = 1

# ---- after you computed R,t (unit t) and triangulated X (arbitrary scale) ----
# Option 1: multi-edge scale (recommended)
known_edges = [
    (1, 2, 115.0),  # לדוגמה: שתי צלעות של 115
    (3, 4, 115.0),
    (0, 1, 130.0),  # וצלע עומק של 130
]
X= triangulate_all(K,R,t, pts1, pts2)
X, s = scale_from_multiple_edges(X, known_edges)
print(f"scale factor s = {s:.4f}")

# Check orthogonality (sanity)
i0,iX,iY,iZ = EDGE_TRIPLET
v1 = X[iX]-X[i0]; v2 = X[iY]-X[i0]; v3 = X[iZ]-X[i0]
for a,b,name in [(v1,v2,"v1·v2"),(v1,v3,"v1·v3"),(v2,v3,"v2·v3")]:
    cosang = abs(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
    print(name, "cos≈", cosang)

# Robust volume from triple product
vol_mm3 = abs(np.dot(v1, np.cross(v2, v3)))
print("Lengths (mm):", np.linalg.norm(v1), np.linalg.norm(v2), np.linalg.norm(v3))
print("Volume (cm^3):", vol_mm3/1000.0)


# # 2) Triangulate with unscaled t
# X = triangulate_all(K, R, t, pts1, pts2)  # mm only after we scale below

# # 3) Set metric scale
# if (BASELINE_MM is None) ^ (EDGE_LEN_MM is None):
#     if BASELINE_MM is not None:
#         # Scale using baseline: scale 3D points and translation uniformly
#         X *= BASELINE_MM
#     else:
#         # Scale using one known edge length
#         X = scale_by_known_edge(X, EDGE_IDX, EDGE_LEN_MM)
# else:
#     raise RuntimeError("Set exactly one of BASELINE_MM or EDGE_LEN_MM (not both).")

# # 4) Compute volume from three edges meeting at one corner
# vol_mm3, (Lx, Ly, Lz) = volume_from_edges(X, EDGE_TRIPLET)
# print(f"Edge lengths (mm): {Lx:.2f}, {Ly:.2f}, {Lz:.2f}")
# print(f"Estimated volume: {vol_mm3/1000.0:.2f} cm^3")


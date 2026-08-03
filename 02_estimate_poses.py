"""
Step 2: checkerboard PnP -> metric camera-center anchors for each sequence.

These anchors are NOT the per-frame poses used in the final dataset -- they
exist only to give COLMAP's sparse reconstruction a metric scale and a
well-defined world frame (chessboard frame) in step 3 (colmap model_aligner).

Expects, for each sequence folder under --root:
  <root>/<date>/<seq_id>/calib_color.npz   (cameraMatrix, distCoeffs from a
                                             standard OpenCV checkerboard
                                             calibration, done once per camera)
  <root>/<date>/<seq_id>/image/*.png       (from 01_extract_frames.py)

Writes, per sequence:
  <root>/<date>/<seq_id>/poses/distances_image.csv
  <root>/<date>/<seq_id>/poses/poses_image.npz
  <root>/<date>/<seq_id>/poses/refs.txt     (camera centers, consumed by
                                              `colmap model_aligner
                                              --ref_images_path refs.txt`)
"""
import argparse
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np

PATTERN_SIZE = (10, 7)
SQUARE_SIZE = 0.025  # meters


def to_colmap_image_name(img_path: str) -> str:
    stem = Path(img_path).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits == "":
        raise ValueError(f"No digits found in filename stem: {img_path}")
    return f"{int(digits):06d}.png"


def process_folder(root_folder: str):
    calib_path = os.path.join(root_folder, "calib_color.npz")
    images_dir = os.path.join(root_folder, "image")
    if not os.path.exists(calib_path) or not os.path.isdir(images_dir):
        print(f"[SKIP] Missing calib_color.npz or image/ in {root_folder}")
        return

    calib_data = np.load(calib_path)
    camera_matrix = calib_data["cameraMatrix"]
    dist_coeffs = calib_data["distCoeffs"]
    if dist_coeffs.ndim == 1:
        dist_coeffs = dist_coeffs.reshape(1, -1)

    w, h = PATTERN_SIZE
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2) * SQUARE_SIZE

    output_dir = os.path.join(root_folder, "poses")
    os.makedirs(output_dir, exist_ok=True)

    img_exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    img_paths = []
    for ext in img_exts:
        img_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    def sort_key(p):
        digs = "".join(ch for ch in Path(p).stem if ch.isdigit())
        return int(digs) if digs else 10**18

    img_paths = sorted(img_paths, key=sort_key)
    print(f"[{root_folder}] {len(img_paths)} images")

    distances, all_rvecs, all_tvecs, image_names = [], [], [], []
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK
        )
        if not found:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        success, rvec, tvec = cv2.solvePnP(objp, corners.reshape(-1, 1, 2), camera_matrix, dist_coeffs)
        if not success:
            continue
        try:
            image_name = to_colmap_image_name(img_path)
        except ValueError:
            continue
        distances.append((image_name, float(np.linalg.norm(tvec))))
        all_rvecs.append(rvec)
        all_tvecs.append(tvec)
        image_names.append(image_name)

    with open(os.path.join(output_dir, "distances_image.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "distance_m"])
        writer.writerows(distances)

    np.savez_compressed(
        os.path.join(output_dir, "poses_image.npz"),
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs,
        rvecs=np.array(all_rvecs), tvecs=np.array(all_tvecs),
        distances=np.array([d[1] for d in distances], dtype=float),
        image_names=np.array(image_names, dtype=object),
    )

    if len(all_tvecs) >= 3:
        cam_centers = []
        for rvec, tvec in zip(np.array(all_rvecs).reshape(-1, 3), np.array(all_tvecs).reshape(-1, 3)):
            R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            cam_centers.append((-R.T @ tvec.reshape(3, 1)).reshape(3))
        with open(os.path.join(output_dir, "refs.txt"), "w") as f:
            for name, C in zip(image_names, cam_centers):
                f.write(f"{name} {C[0]:.6f} {C[1]:.6f} {C[2]:.6f}\n")
        print(f"[{root_folder}] {len(image_names)} checkerboard-visible anchor frames")
    else:
        print(f"[{root_folder}] WARNING: fewer than 3 checkerboard-visible frames, "
              f"model_aligner will fail in step 3")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True,
                    help="Root folder containing <date>/<seq_id>/{calib_color.npz,image/}")
    args = p.parse_args()

    root_path = Path(args.root)
    subfolders = [d for d in root_path.rglob("*")
                  if d.is_dir() and (d / "calib_color.npz").exists() and (d / "image").is_dir()]
    print(f"Found {len(subfolders)} sequence folders")
    for sub in sorted(subfolders):
        process_folder(str(sub))
    print("Done.")

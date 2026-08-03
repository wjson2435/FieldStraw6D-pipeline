"""
Step 5: assemble the final per-plant dataset from:
  - COLMAP's undistorted, metrically-aligned reconstruction (step 3)
  - a labelCloud 3D box annotation on the sequence's Poisson mesh (step 4a)
  - a CVAT (YOLO-format) 2D bbox annotation on the sequence's frames (step 4b)

This produces, per physical plant (one reconstructed sequence == one plant):

  <output_root>/<plant_id>/000000.png, 000001.png, ...
  <output_root>/<plant_id>/metadata.jsonl   (one row per image, HF imagefolder convention)

matching the schema used by the published Straw6D dataset on HuggingFace.
(If you're assembling train/validation/test splits afterwards, HF's
imagefolder loader expects one metadata.jsonl per split rather than per
plant folder -- merge the per-plant files into a split-level one, with
file_name rewritten to "<plant_id>/<image>.png", before uploading.)

Every per-frame conversion below (labelCloud's centroid/dimensions/rotations
-> local_to_world_transform, YOLO bbox -> pixel bbox, COLMAP pose ->
camera_view_matrix, intrinsics -> camera_projection_matrix) was reverse-
derived from and validated bit-exact against that released dataset -- see
the comments inline for the exact matrices and where each one comes from.

A frame is kept once it has at least one CVAT detection clearing
--min_bbox_area/--border_margin (see below); this is a simple, tunable rule,
not an attempt to reproduce any particular curation pass a human annotator
might have layered on top for a specific dataset release.

Expected input layout, matching steps 1-4:

  <data_root>/<date>/<seq_id>/
    dense/sparse/{cameras.bin, images.bin}   # undistorted COLMAP model (step 3)
    dense/images/*.png                       # undistorted frames (step 3)
    annotation/*/obj_train_data/*.txt        # CVAT YOLO 2D bbox export (step 4b)

  <labels_root>/<date>_<seq_id>_mesh_poisson.json   # labelCloud export (step 4a)

A sequence is skipped (with a warning) if reconstruction, annotation, or
2D-bbox export is missing/incomplete for it -- not every recorded sequence
survives to the final dataset (see the paper's dataset-analysis section).

Some sequences contain more than one fruit (2-4 labelCloud boxes, matching
multi-line CVAT YOLO frames). Since neither annotation tool shares an object
ID with the other, each frame's YOLO detections are matched to labelCloud
boxes by projecting every box's world centroid into that frame with the
COLMAP pose + intrinsics and taking the nearest 2D detection (linear_sum_
assignment) -- validated on real multi-object frames, where the correct
pairing is unambiguous (~50px to the right box vs. ~200px+ to the wrong one).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from utils.colmap_io import read_cameras_binary, read_images_binary, qvec2rotmat

# Axis-swap fixed matrix mapping labelCloud's local box frame (x=length,
# y=width, z=height) to the dataset's local frame (x=height, y=width,
# z=length). Combined with a transpose, this exactly reproduces the
# local_to_world_transform rotation already published in the dataset
# (validated to <1e-15 max abs error across every sequence checked).
_AXIS_SWAP = np.array([[0., 0., 1.],
                        [0., 1., 0.],
                        [1., 0., 0.]])

# COLMAP (OpenCV) camera axes -> the dataset's camera axes (Y and Z flipped),
# i.e. the standard OpenCV-camera -> OpenGL/USD-camera convention change.
_CAM_AXIS_FLIP = np.diag([1., -1., -1.])


def labelcloud_obj_to_pose(obj: dict):
    """labelCloud object annotation -> (local_to_world_transform, size_local)."""
    c, dims, rot = obj["centroid"], obj["dimensions"], obj["rotations"]
    R = Rotation.from_euler("xyz", [rot["x"], rot["y"], rot["z"]]).as_matrix()
    R_final = _AXIS_SWAP @ R.T

    T = np.eye(4)
    T[:3, :3] = R_final
    T[3, :3] = [c["x"], c["y"], c["z"]]

    size_local = [dims["height"], dims["width"], dims["length"]]
    return T.tolist(), size_local


def colmap_pose_to_view_matrix(qvec: np.ndarray, tvec: np.ndarray) -> list:
    """COLMAP world-to-camera (qvec, tvec) -> camera_view_matrix (row-vector, 4x4)."""
    R = qvec2rotmat(qvec)
    V = np.eye(4)
    V[:3, :3] = R.T @ _CAM_AXIS_FLIP
    V[3, :3] = _CAM_AXIS_FLIP @ tvec
    return V.tolist()


def build_projection_matrix(fx: float, fy: float, width: float, height: float, near_plane: float) -> list:
    P = np.zeros((4, 4))
    P[0, 0] = 2.0 * fx / width
    P[1, 1] = 2.0 * fy / height
    P[2, 3] = -1.0
    P[3, 2] = near_plane
    return P.tolist()


def load_yolo_bboxes(txt_path: Path, raw_w: int, raw_h: int):
    """CVAT YOLO export (one 'class cx cy w h' line per detected fruit, normalized)
    -> list of [xmin, ymin, xmax, ymax] in raw pixel coordinates."""
    boxes = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        _, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        boxes.append([
            (cx - w / 2) * raw_w, (cy - h / 2) * raw_h,
            (cx + w / 2) * raw_w, (cy + h / 2) * raw_h,
        ])
    return boxes


def find_yolo_dir(seq_root: Path):
    matches = list(seq_root.glob("annotation/*/obj_train_data"))
    return matches[0] if matches else None


def load_label_objects(labels_root: Path, date: str, seq_id: str):
    """labelCloud can export a sequence's boxes under either the fused- or
    mesh_poisson-reconstruction name; use whichever one actually has objects."""
    for suffix in ("mesh_poisson", "fused"):
        path = labels_root / f"{date}_{seq_id}_{suffix}.json"
        if not path.exists():
            continue
        objects = json.loads(path.read_text()).get("objects", [])
        if objects:
            return objects
    return []


def project_point(world_xyz: np.ndarray, R: np.ndarray, t: np.ndarray, fx, fy, cx, cy):
    """World point -> pixel (u, v) using COLMAP's raw world-to-camera (R, t) + pinhole intrinsics."""
    p_cam = R @ world_xyz + t
    return np.array([fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy])


def match_boxes_to_detections(world_centroids, yolo_boxes, R, t, fx, fy, cx, cy, max_dist):
    """Assign each YOLO detection to its nearest projected 3D-box centroid.
    Returns a list of (label_object_index, bbox) pairs, one per YOLO detection
    that has a plausible match (guards against spurious/occluded detections)."""
    if not yolo_boxes:
        return []
    projected = np.array([project_point(c, R, t, fx, fy, cx, cy) for c in world_centroids])
    det_centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in yolo_boxes])
    cost = np.linalg.norm(projected[:, None, :] - det_centers[None, :, :], axis=2)
    obj_idx, det_idx = linear_sum_assignment(cost)
    pairs = []
    for oi, di in zip(obj_idx, det_idx):
        if cost[oi, di] < max_dist:
            pairs.append((oi, yolo_boxes[di]))
    return pairs


def build_sequence(seq_root: Path, label_objects: list, out_root: Path, plant_id: str, cfg):
    dense_sparse = seq_root / "dense" / "sparse"
    images_dir = seq_root / "dense" / "images"
    yolo_dir = find_yolo_dir(seq_root)

    if not (dense_sparse / "images.bin").exists():
        print(f"[SKIP] {seq_root}: no undistorted COLMAP model (run step 3)")
        return 0
    if yolo_dir is None:
        print(f"[SKIP] {seq_root}: no CVAT YOLO export under annotation/*/obj_train_data")
        return 0

    # one entry per fruit in this sequence (usually 1, occasionally 2-4)
    obj_poses = [labelcloud_obj_to_pose(o) for o in label_objects]
    world_centroids = [np.array([o["centroid"]["x"], o["centroid"]["y"], o["centroid"]["z"]])
                        for o in label_objects]

    colmap_images = read_images_binary(dense_sparse / "images.bin")
    colmap_cameras = read_cameras_binary(dense_sparse / "cameras.bin")
    by_name = {im["name"]: im for im in colmap_images.values()}

    out_dir = out_root / plant_id
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for yolo_txt in sorted(yolo_dir.glob("*.txt")):
        stem = yolo_txt.stem
        img_name = f"{stem}.png"
        undistorted_img = images_dir / img_name
        if img_name not in by_name or not undistorted_img.exists():
            continue

        yolo_boxes = load_yolo_bboxes(yolo_txt, cfg.raw_frame_width, cfg.raw_frame_height)
        if not yolo_boxes:
            continue  # no object visible in this frame

        colmap_im = by_name[img_name]
        cam = colmap_cameras[colmap_im["camera_id"]]
        params = cam["params"]
        if len(params) == 3:  # SIMPLE_PINHOLE / SIMPLE_RADIAL: (f, cx, cy)
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # PINHOLE and others: (fx, fy, cx, cy, ...)
            fx, fy, cx, cy = params[:4]
        width, height = cam["width"], cam["height"]

        R = qvec2rotmat(colmap_im["qvec"])
        pairs = match_boxes_to_detections(world_centroids, yolo_boxes, R, colmap_im["tvec"],
                                           fx, fy, cx, cy, cfg.match_max_dist)
        if not pairs:
            continue

        camera_view_matrix = colmap_pose_to_view_matrix(colmap_im["qvec"], colmap_im["tvec"])
        camera_projection_matrix = build_projection_matrix(fx, fy, width, height, cfg.near_plane)

        labels, bboxes, sizes, centers, transforms = [], [], [], [], []
        for obj_idx, bbox in pairs:
            x1, y1, x2, y2 = bbox
            if (x2 - x1) * (y2 - y1) < cfg.min_bbox_area:
                continue
            if x1 <= cfg.border_margin or y1 <= cfg.border_margin or \
               x2 >= width - cfg.border_margin or y2 >= height - cfg.border_margin:
                continue
            local_to_world_transform, size_local = obj_poses[obj_idx]
            labels.append("strawberry")
            bboxes.append(bbox)
            sizes.append(size_local)
            centers.append([0.0, 0.0, 0.0])
            transforms.append(local_to_world_transform)
        if not labels:
            continue  # every detection in this frame was filtered out

        shutil.copy2(undistorted_img, out_dir / img_name)
        rows.append({
            "file_name": img_name,
            "image_id": stem,
            "resolution": [width, height],
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "camera_view_matrix": camera_view_matrix,
            "camera_projection_matrix": camera_projection_matrix,
            "labels": labels,
            "bbox_2d_loose": bboxes,
            "size_local": sizes,
            "center_local": centers,
            "local_to_world_transform": transforms,
        })

    if not rows:
        shutil.rmtree(out_dir, ignore_errors=True)
        return 0

    with open(out_dir / "metadata.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="Root containing <date>/<seq_id>/ sequence folders (steps 1-3 output)")
    p.add_argument("--labels_root", type=str, required=True,
                    help="Folder of labelCloud exports: <date>_<seq_id>_mesh_poisson.json")
    p.add_argument("--output_root", type=str, required=True,
                    help="Where to write <plant_id>/ folders")
    p.add_argument("--raw_frame_width", type=int, default=640,
                    help="Pre-undistortion frame width the CVAT YOLO boxes were drawn on")
    p.add_argument("--raw_frame_height", type=int, default=480,
                    help="Pre-undistortion frame height the CVAT YOLO boxes were drawn on")
    p.add_argument("--min_bbox_area", type=float, default=200,
                    help="px^2; drop degenerate/near-empty detections")
    p.add_argument("--border_margin", type=float, default=1,
                    help="px; drop detections clipped by the frame edge")
    p.add_argument("--near_plane", type=float, default=0.05,
                    help="Near-plane constant used in the projection matrix")
    p.add_argument("--match_max_dist", type=float, default=150,
                    help="px; max distance between a projected 3D-box centroid and a YOLO "
                         "detection for them to be considered the same object")
    args = p.parse_args()

    data_root = Path(args.data_root)
    labels_root = Path(args.labels_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(d for d in data_root.glob("*/*") if d.is_dir())
    plant_counter = 0
    total_frames = 0
    for seq_root in seq_dirs:
        date, seq_id = seq_root.parent.name, seq_root.name
        label_objects = load_label_objects(labels_root, date, seq_id)
        if not label_objects:
            continue  # not annotated (or annotated with zero fruit) -> not a candidate plant

        plant_counter += 1
        plant_id = f"plant_{plant_counter:03d}"
        n = build_sequence(seq_root, label_objects, output_root, plant_id, args)
        if n > 0:
            print(f"[OK] {date}/{seq_id} -> {plant_id}: {n} frames, "
                  f"{len(label_objects)} fruit")
            total_frames += n
        else:
            plant_counter -= 1  # didn't actually produce a plant folder

    print(f"\nDone. {plant_counter} plants, {total_frames} total frames written to {output_root}")


if __name__ == "__main__":
    main()

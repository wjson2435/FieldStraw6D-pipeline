# Straw6D data-collection pipeline

Scripts used to build the real-world portion of [Straw6D](https://huggingface.co/datasets/WoojungSon/Straw6D):
in-field video of a strawberry plant -> metrically-scaled 3D reconstruction ->
manual 3D/2D annotation -> a per-plant dataset folder, in the exact schema
published on HuggingFace.

## Requirements

- Python 3.10+, `pip install -r requirements.txt`
- [COLMAP](https://colmap.github.io/) on `PATH` (step 3)
- [labelCloud](https://github.com/ch-sa/labelCloud) (step 4b, 3D box annotation)
- Any tool that can export YOLO-format 2D bounding boxes (step 4c), e.g. [CVAT](https://github.com/cvat-ai/cvat)
- A checkerboard-calibrated RGB camera: run [OpenCV's standard camera
  calibration routine](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
  once per camera and save the result as `calib_color.npz` with keys
  `cameraMatrix` (3x3) and `distCoeffs` (1x5)

## Data layout

Three root folders; everything below is produced by the step that's marked.

```
<data_root>/<date>/<seq_id>/       one folder per recorded plant
  rgb_raw.mp4, calib_color.npz       (input)
  image/                             step 1
  poses/                             step 2
  sparse/0/, sparse/sparse_align/    step 3 (raw / metrically-aligned SfM)
  dense/                             step 3 (undistorted images, camera model, mesh_poisson.ply)
  annotation/*.txt                   step 4c (manual, 2D boxes)

<pointcloud_folder>/<date>_<seq_id>_mesh_poisson.ply   step 4a (gathered meshes, labelCloud input)
<labels_root>/<date>_<seq_id>_mesh_poisson.json        step 4b (manual, 3D box, labelCloud output)

<output_root>/plant_XXX/           step 5 (final dataset, one folder per plant)
  rgb/000000.png, ...
  json/000000.json, ...
```

## Pipeline

Each physical plant is recorded as one video clip:

```
<data_root>/<date>/<seq_id>/
  rgb_raw.mp4
  calib_color.npz
```

**1. Extract frames** -> `<seq_id>/image/000000.png, ...`
```
python 01_extract_frames.py --root <data_root> [--frame_skip 5] [--cut_seconds 1.0]
```

**2. Checkerboard PnP anchors** -> `<seq_id>/poses/refs.txt`
```
python 02_estimate_poses.py --root <data_root> \
    [--pattern_cols 10] [--pattern_rows 7] [--square_size 0.025]
```
Gives step 3 a metric scale and world frame; not the final per-frame poses.

**3. COLMAP reconstruction + metric alignment** -> `<seq_id>/dense/` (undistorted images, camera model, `mesh_poisson.ply`)
```
./03_reconstruct.sh --root <data_root> [--only <date>/<seq_id>]
```
Runs every sequence that has `image/` + `poses/refs.txt`; a sequence that
fails is logged and skipped rather than aborting the batch. `--only` reruns
a single sequence.

**4. Annotate**

- **4a. Gather meshes for labelCloud** -> `<pointcloud_folder>/<date>_<seq_id>_mesh_poisson.ply`
  ```
  python 04_gather_pointclouds.py --data_root <data_root> --pointcloud_folder <pointcloud_folder> [--symlink]
  ```
  labelCloud annotates every point cloud in one configured folder and has
  no notion of our `<date>/<seq_id>/` layout, so this collects each
  sequence's `dense/mesh_poisson.ply` into one flat folder first.

- **4b. 3D box (manual)** -- open `<pointcloud_folder>` in labelCloud
  (`configs/labelcloud_config.ini`), draw one box per fruit per point
  cloud, with `label_folder` set to `<labels_root>` -> one
  `<date>_<seq_id>_mesh_poisson.json` per point cloud.

- **4c. 2D box (manual)** -- per frame on `<seq_id>/image/*.png`, any tool,
  exported as YOLO -> `<seq_id>/annotation/<frame_stem>.txt` (flat; CVAT's
  default nested `annotation/<task>/obj_train_data/` export is also
  auto-detected). One `class cx cy w h` line per visible fruit,
  coordinates normalized [0,1] to the **raw** (pre-undistortion, step-1)
  frame size -- class index is ignored (single category).

**5. Build the final dataset** -> `<output_root>/plant_001/{rgb/000000.png, json/000000.json}, ...`
```
python 05_build_dataset.py \
    --data_root <data_root> --labels_root <labels_root> --output_root <output_root> \
    [--raw_frame_width 640] [--raw_frame_height 480] \
    [--min_bbox_area 200] [--border_margin 1] [--near_plane 0.05] [--match_max_dist 150]
```
One `plant_XXX` folder per sequence, each image paired with its own
`{camera_data, objects}` json. Splitting into train/val/test is left to the
user -- the published dataset uses a plant-disjoint split, which is what you
should replicate.

See the docstring at the top of `05_build_dataset.py` for exactly how each
field is derived (and validated against the released dataset).

## Full example

```bash
DATA_ROOT=~/straw6d_raw
POINTCLOUD_FOLDER=~/straw6d_pointclouds
LABELS_ROOT=~/straw6d_labels
OUTPUT_ROOT=~/straw6d_final

python 01_extract_frames.py --root $DATA_ROOT
python 02_estimate_poses.py --root $DATA_ROOT
./03_reconstruct.sh --root $DATA_ROOT
python 04_gather_pointclouds.py --data_root $DATA_ROOT --pointcloud_folder $POINTCLOUD_FOLDER
# -- annotate: labelCloud on $POINTCLOUD_FOLDER (-> $LABELS_ROOT), YOLO 2D boxes (step 4c) --
python 05_build_dataset.py --data_root $DATA_ROOT --labels_root $LABELS_ROOT --output_root $OUTPUT_ROOT
```

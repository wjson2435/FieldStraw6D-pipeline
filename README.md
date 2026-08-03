# Straw6D data-collection pipeline

Scripts used to build the real-world portion of [Straw6D](https://huggingface.co/datasets/WoojungSon/Straw6D):
in-field video of a strawberry plant -> metrically-scaled 3D reconstruction ->
manual 3D/2D annotation -> a per-plant dataset folder, in the exact schema
published on HuggingFace.

## Requirements

- Python 3.10+, `pip install -r requirements.txt`
- [COLMAP](https://colmap.github.io/) on `PATH` (step 3)
- [labelCloud](https://github.com/ch-sa/labelCloud) (step 4a, 3D box annotation)
- Any tool that can export YOLO-format 2D bounding boxes (step 4b), e.g. [CVAT](https://github.com/cvat-ai/cvat)
- A checkerboard-calibrated RGB camera: run [OpenCV's standard camera
  calibration routine](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
  once per camera and save the result as `calib_color.npz` with keys
  `cameraMatrix` (3x3) and `distCoeffs` (1x5)

## Pipeline

Each physical plant is recorded as one video clip (`rgb_raw.mp4`), organized as:

```
<data_root>/<date>/<seq_id>/
  rgb_raw.mp4
  calib_color.npz
```

**1. Extract frames**
```
python 01_extract_frames.py --root <data_root> [--frame_skip 5] [--cut_seconds 1.0]
```
Drops the first/last `--cut_seconds` of each clip and keeps every
`--frame_skip`-th remaining frame -> `<data_root>/<date>/<seq_id>/image/000000.png, ...`

**2. Checkerboard PnP anchors**
```
python 02_estimate_poses.py --root <data_root> \
    [--pattern_cols 10] [--pattern_rows 7] [--square_size 0.025]
```
Finds the checkerboard (`--pattern_cols`x`--pattern_rows` inner corners,
`--square_size` meters per square) in whichever frames it's visible in and
solves PnP to get metric camera centers for those frames. This does **not**
produce the final per-frame poses -- it exists purely to give COLMAP's next
step a metric scale and a common (chessboard) world frame.
-> `<data_root>/<date>/<seq_id>/poses/{distances_image.csv, poses_image.npz, refs.txt}`

**3. COLMAP reconstruction + metric alignment**
```
./03_reconstruct.sh <data_root>/<date>/<seq_id>
```
Feature extraction -> exhaustive matching -> sparse mapping -> `model_aligner`
(fits a similarity transform from step 2's anchors, resolving SfM's scale/
gauge ambiguity) -> undistortion -> dense stereo -> Poisson mesh. Run once
per sequence folder.
-> `<seq>/sparse/sparse_align/` (aligned sparse model), `<seq>/dense/` (undistorted
images + camera model + `mesh_poisson.ply`)

**4. Annotate (manual, not scripted)**

- **3D box** -- open `<seq>/dense/mesh_poisson.ply` in labelCloud
  (`configs/labelcloud_config.ini` has the settings used for this dataset)
  and draw one bounding box per fruit. Export to
  `<labels_root>/<date>_<seq_id>_mesh_poisson.json`.
- **2D box** -- annotate a bounding box per visible fruit per frame in
  `<seq>/image/*.png`, using whatever tool you prefer, and export in
  **YOLO format** (one `class cx cy w h` line per box, normalized
  coordinates) to `<seq>/annotation/<task_name>/obj_train_data/*.txt`.

**5. Build the final dataset**
```
python 05_build_dataset.py \
    --data_root <data_root> \
    --labels_root <labels_root> \
    --output_root <output_root> \
    [--raw_frame_width 640] [--raw_frame_height 480] \
    [--min_bbox_area 200] [--border_margin 1] \
    [--near_plane 0.05] [--match_max_dist 150]
```
For every sequence that has both a 3D annotation and a 2D annotation, this
combines labelCloud's box(es), CVAT's per-frame detections, and COLMAP's
per-frame pose + intrinsics into one dataset per plant:

```
<output_root>/plant_001/000000.png
<output_root>/plant_001/metadata.jsonl
<output_root>/plant_002/...
```

`metadata.jsonl` has one row per image (HF imagefolder convention), with all
pose/camera/bbox fields inline. One `plant_XXX` folder per reconstructed
sequence (one physical plant each). Grouping/splitting these into
train/val/test is left to the user -- the published dataset uses a
plant-disjoint split (no plant's frames appear in more than one of
train/validation/test), which is the split you should replicate if training
against it. If assembling train/val/test splits for upload to HuggingFace,
note that its imagefolder loader expects one `metadata.jsonl` per split
rather than per plant -- merge the per-plant files, rewriting `file_name` to
`<plant_id>/<image>.png`.

See the docstring at the top of `05_build_dataset.py` for exactly how each
field is derived (and validated against the released dataset).

## Full example

All five steps chained together for one recording session (`2026-01-19/000000`):

```bash
DATA_ROOT=~/straw6d_raw
LABELS_ROOT=~/straw6d_labels
OUTPUT_ROOT=~/straw6d_final

# ~/straw6d_raw/2026-01-19/000000/{rgb_raw.mp4, calib_color.npz} already in place

python 01_extract_frames.py --root $DATA_ROOT
python 02_estimate_poses.py --root $DATA_ROOT
./03_reconstruct.sh $DATA_ROOT/2026-01-19/000000

# -- manual: annotate 3D box in labelCloud on
#      $DATA_ROOT/2026-01-19/000000/dense/mesh_poisson.ply
#    export to $LABELS_ROOT/2026-01-19_000000_mesh_poisson.json
# -- manual: annotate 2D boxes (YOLO format) on
#      $DATA_ROOT/2026-01-19/000000/image/*.png
#    export to $DATA_ROOT/2026-01-19/000000/annotation/<task>/obj_train_data/*.txt

python 05_build_dataset.py \
    --data_root $DATA_ROOT \
    --labels_root $LABELS_ROOT \
    --output_root $OUTPUT_ROOT

# -> $OUTPUT_ROOT/plant_001/000000.png, metadata.jsonl, ...
```

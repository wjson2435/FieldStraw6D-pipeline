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
./03_reconstruct.sh <data_root>/<date>/<seq_id>
```

**4. Annotate (manual, not scripted)**
- **3D box** in labelCloud on `<seq_id>/dense/mesh_poisson.ply` (`configs/labelcloud_config.ini`) -> `<labels_root>/<date>_<seq_id>_mesh_poisson.json`
- **2D box** per frame on `<seq_id>/image/*.png`, any tool, exported as YOLO (`class cx cy w h`) -> `<seq_id>/annotation/<task_name>/obj_train_data/*.txt`

**5. Build the final dataset** -> `<output_root>/plant_001/{000000.png, ..., metadata.jsonl}`
```
python 05_build_dataset.py \
    --data_root <data_root> --labels_root <labels_root> --output_root <output_root> \
    [--raw_frame_width 640] [--raw_frame_height 480] \
    [--min_bbox_area 200] [--border_margin 1] [--near_plane 0.05] [--match_max_dist 150]
```
One `plant_XXX` folder per sequence; `metadata.jsonl` has one row per image
(HF imagefolder convention). Splitting into train/val/test is left to the
user -- the published dataset uses a plant-disjoint split, which is what you
should replicate. HF's imagefolder loader expects one `metadata.jsonl` per
split rather than per plant, so merge the per-plant files first (rewriting
`file_name` to `<plant_id>/<image>.png`).

See the docstring at the top of `05_build_dataset.py` for exactly how each
field is derived (and validated against the released dataset).

## Full example

```bash
DATA_ROOT=~/straw6d_raw
LABELS_ROOT=~/straw6d_labels
OUTPUT_ROOT=~/straw6d_final

python 01_extract_frames.py --root $DATA_ROOT
python 02_estimate_poses.py --root $DATA_ROOT
./03_reconstruct.sh $DATA_ROOT/2026-01-19/000000
# -- annotate (step 4) --
python 05_build_dataset.py --data_root $DATA_ROOT --labels_root $LABELS_ROOT --output_root $OUTPUT_ROOT
```

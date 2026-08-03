# Straw6D data-collection pipeline

Scripts used to build the real-world portion of [SonUF/Straw6D](https://huggingface.co/datasets/SonUF/Straw6D):
in-field video of a strawberry plant -> metrically-scaled 3D reconstruction ->
manual 3D/2D annotation -> a per-plant `rgb/` + `json/` dataset folder, in the
exact schema published on HuggingFace.

This is the **data pipeline only**. Training/evaluation code lives in a
separate repo.

## Requirements

- Python 3.10+, `pip install -r requirements.txt`
- [COLMAP](https://colmap.github.io/) on `PATH` (step 3)
- [labelCloud](https://github.com/ch-sa/labelCloud) (step 4a, 3D box annotation)
- [CVAT](https://github.com/cvat-ai/cvat) self-hosted, or any tool that can
  export YOLO-format 2D bounding boxes (step 4b)
- A checkerboard-calibrated RGB camera (`cameraMatrix` / `distCoeffs` per the
  standard OpenCV calibration routine, saved as `calib_color.npz`)

## Pipeline

Each physical plant is recorded as one video clip (`rgb_raw.mp4`), organized as:

```
<data_root>/<date>/<seq_id>/
  rgb_raw.mp4
  calib_color.npz
```

**1. Extract frames**
```
python 01_extract_frames.py --root <data_root>
```
Drops the first/last second of each clip and keeps every 5th remaining frame
-> `<data_root>/<date>/<seq_id>/image/000000.png, ...`

**2. Checkerboard PnP anchors**
```
python 02_estimate_poses.py --root <data_root>
```
Finds the checkerboard in whichever frames it's visible in and solves PnP to
get metric camera centers for those frames. This does **not** produce the
final per-frame poses -- it exists purely to give COLMAP's next step a
metric scale and a common (chessboard) world frame.
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
- **2D box** -- import `<seq>/image/*.png` into CVAT, draw/track a bounding
  box per visible fruit per frame, export as **YOLO 1.1** ->
  `<seq>/annotation/<task_name>/obj_train_data/*.txt`.

**5. Build the final dataset**
```
python 05_build_dataset.py \
    --data_root <data_root> \
    --labels_root <labels_root> \
    --output_root <output_root>
```
For every sequence that has both a 3D annotation and a 2D annotation, this
combines labelCloud's box(es), CVAT's per-frame detections, and COLMAP's
per-frame pose + intrinsics into one JSON per frame, and copies the matching
undistorted image alongside it:

```
<output_root>/plant_001/rgb/000000.png
<output_root>/plant_001/json/000000.json
<output_root>/plant_002/...
```

One `plant_XXX` folder per reconstructed sequence (one physical plant each).
Grouping/splitting these into train/val/test is left to the user --
`SonUF/Straw6D` uses a plant-disjoint split (no plant's frames appear in more
than one of train/validation/test), which is the split you should replicate
if training against the released dataset.

**Frame selection.** A frame is kept once it has one usable CVAT detection
(min area + not clipped by the image border, see `MIN_BBOX_AREA`/
`BORDER_MARGIN` in the script). This is a superset of what shipped in
`SonUF/Straw6D` -- the original release applied one further manual/QC
curation pass on top (mainly visible in multi-fruit sequences) that isn't
reproduced here. Every field this script does compute is validated
bit-exact against the release; it's only the "would a human have kept this
particular frame" judgment call that isn't reconstructed.

See the docstring at the top of `05_build_dataset.py` for exactly how each
field is derived (and validated against the released dataset).

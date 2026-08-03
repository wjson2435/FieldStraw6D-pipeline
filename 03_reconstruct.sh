#!/bin/bash
# Step 3: COLMAP sparse reconstruction, metric alignment, and dense mesh.
#
# Usage: ./03_reconstruct.sh <sequence_path>
#   <sequence_path> must contain:
#     image/            (from 01_extract_frames.py)
#     poses/refs.txt    (from 02_estimate_poses.py -- metric anchor frames)
#
# The `colmap model_aligner` step is what resolves SfM's inherent scale/gauge
# ambiguity: it fits a similarity transform (rotation + scale + translation)
# that best matches COLMAP's own camera centers for the anchor frames in
# refs.txt to their metric, chessboard-frame positions from step 2. Every
# frame's pose is then expressed in that same metric chessboard world frame.
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <sequence_path>"
    exit 1
fi

PROJECT_PATH="$1"
DATABASE_PATH="$PROJECT_PATH/database.db"
IMAGE_PATH="$PROJECT_PATH/image"
SPARSE_PATH="$PROJECT_PATH/sparse"

mkdir -p "$SPARSE_PATH"

echo "Extracting features..."
colmap feature_extractor \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH"

echo "Matching features (exhaustive)..."
colmap exhaustive_matcher \
    --database_path "$DATABASE_PATH"

echo "Running sparse reconstruction (mapper)..."
colmap mapper \
    --database_path "$DATABASE_PATH" \
    --image_path "$IMAGE_PATH" \
    --output_path "$SPARSE_PATH"

SPARSE_PATH_OUTPUT="$PROJECT_PATH/sparse/0"
REF_POSES_PATH="$PROJECT_PATH/poses/refs.txt"
ALIGNED_SPARSE="$PROJECT_PATH/sparse/sparse_align"
mkdir -p "$ALIGNED_SPARSE"

echo "Aligning model to metric world frame using checkerboard-PnP anchors..."
colmap model_aligner \
    --input_path "$SPARSE_PATH_OUTPUT" \
    --output_path "$ALIGNED_SPARSE" \
    --ref_images_path "$REF_POSES_PATH" \
    --ref_is_gps 0 \
    --alignment_max_error 3.0

DENSE_WS="$PROJECT_PATH/dense"
mkdir -p "$DENSE_WS"

echo "Undistorting images..."
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$ALIGNED_SPARSE" \
    --output_path "$DENSE_WS" \
    --output_type COLMAP

echo "1) PatchMatch stereo..."
colmap patch_match_stereo \
  --workspace_path "$DENSE_WS" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true \
  --PatchMatchStereo.filter true

echo "2) Stereo fusion -> dense point cloud..."
colmap stereo_fusion \
  --workspace_path "$DENSE_WS" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$DENSE_WS/fused.ply"

echo "3) Poisson meshing..."
colmap poisson_mesher \
  --input_path "$DENSE_WS/fused.ply" \
  --output_path "$DENSE_WS/mesh_poisson.ply"

echo "Done."
echo "  Aligned sparse model (poses, used in step 5): $ALIGNED_SPARSE"
echo "  Undistorted camera model (intrinsics, used in step 5): $DENSE_WS/sparse"
echo "  Dense point cloud:  $DENSE_WS/fused.ply"
echo "  Poisson mesh (annotate this in labelCloud, step 4): $DENSE_WS/mesh_poisson.ply"

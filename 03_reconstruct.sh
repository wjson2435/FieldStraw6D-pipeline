#!/bin/bash
# Step 3: COLMAP sparse reconstruction, metric alignment, and dense mesh,
# run over every sequence under <data_root>.
#
# Usage: ./03_reconstruct.sh --root <data_root> [--only <date>/<seq_id>]
#   Each <data_root>/<date>/<seq_id>/ must contain:
#     image/            (from 01_extract_frames.py)
#     poses/refs.txt    (from 02_estimate_poses.py -- metric anchor frames)
#   --only restricts the run to a single sequence (e.g. to retry a failure).
#
# The `colmap model_aligner` step is what resolves SfM's inherent scale/gauge
# ambiguity: it fits a similarity transform (rotation + scale + translation)
# that best matches COLMAP's own camera centers for the anchor frames in
# refs.txt to their metric, chessboard-frame positions from step 2. Every
# frame's pose is then expressed in that same metric chessboard world frame.
#
# A sequence that fails (e.g. too few feature matches) is logged and skipped
# rather than aborting the whole batch.

DATA_ROOT=""
ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --root) DATA_ROOT="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
if [ -z "$DATA_ROOT" ]; then
    echo "Usage: $0 --root <data_root> [--only <date>/<seq_id>]"
    exit 1
fi

reconstruct_one () {
    local PROJECT_PATH="$1"
    local DATABASE_PATH="$PROJECT_PATH/database.db"
    local IMAGE_PATH="$PROJECT_PATH/image"
    local SPARSE_PATH="$PROJECT_PATH/sparse"

    mkdir -p "$SPARSE_PATH"

    colmap feature_extractor --database_path "$DATABASE_PATH" --image_path "$IMAGE_PATH" &&
    colmap exhaustive_matcher --database_path "$DATABASE_PATH" &&
    colmap mapper --database_path "$DATABASE_PATH" --image_path "$IMAGE_PATH" --output_path "$SPARSE_PATH"
    if [ ! -d "$SPARSE_PATH/0" ]; then
        echo "[FAIL] $PROJECT_PATH: sparse mapping produced no model"
        return 1
    fi

    local ALIGNED_SPARSE="$PROJECT_PATH/sparse/sparse_align"
    mkdir -p "$ALIGNED_SPARSE"
    colmap model_aligner \
        --input_path "$SPARSE_PATH/0" \
        --output_path "$ALIGNED_SPARSE" \
        --ref_images_path "$PROJECT_PATH/poses/refs.txt" \
        --ref_is_gps 0 \
        --alignment_max_error 3.0
    if [ ! -f "$ALIGNED_SPARSE/images.bin" ]; then
        echo "[FAIL] $PROJECT_PATH: model_aligner failed (too few checkerboard-visible anchor frames?)"
        return 1
    fi

    local DENSE_WS="$PROJECT_PATH/dense"
    mkdir -p "$DENSE_WS"
    colmap image_undistorter --image_path "$IMAGE_PATH" --input_path "$ALIGNED_SPARSE" \
        --output_path "$DENSE_WS" --output_type COLMAP &&
    colmap patch_match_stereo --workspace_path "$DENSE_WS" --workspace_format COLMAP \
        --PatchMatchStereo.geom_consistency true --PatchMatchStereo.filter true &&
    colmap stereo_fusion --workspace_path "$DENSE_WS" --workspace_format COLMAP \
        --input_type geometric --output_path "$DENSE_WS/fused.ply" &&
    colmap poisson_mesher --input_path "$DENSE_WS/fused.ply" --output_path "$DENSE_WS/mesh_poisson.ply"
    if [ ! -f "$DENSE_WS/mesh_poisson.ply" ]; then
        echo "[FAIL] $PROJECT_PATH: dense reconstruction did not produce mesh_poisson.ply"
        return 1
    fi

    echo "[OK] $PROJECT_PATH -> $DENSE_WS/mesh_poisson.ply (annotate this in labelCloud, step 4)"
}

if [ -n "$ONLY" ]; then
    seq_dirs=("$DATA_ROOT/$ONLY")
else
    seq_dirs=()
    while IFS= read -r d; do seq_dirs+=("$d"); done < <(
        find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d \
            -exec test -d '{}/image' -a -f '{}/poses/refs.txt' \; -print | sort
    )
fi

echo "Reconstructing ${#seq_dirs[@]} sequence(s)..."
n_ok=0
n_fail=0
for seq_dir in "${seq_dirs[@]}"; do
    echo "=== $seq_dir ==="
    if reconstruct_one "$seq_dir"; then
        n_ok=$((n_ok + 1))
    else
        n_fail=$((n_fail + 1))
    fi
done

echo ""
echo "Done. $n_ok succeeded, $n_fail failed/skipped."

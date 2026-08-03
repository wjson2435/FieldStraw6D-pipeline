"""
Step 4a (part 1): gather each sequence's reconstructed mesh into one shared
folder for labelCloud.

labelCloud annotates every point cloud in a single configured
`pointcloud_folder` (see configs/labelcloud_config.ini) and writes one
matching-named json per point cloud into a single `label_folder` -- it has
no notion of our <data_root>/<date>/<seq_id>/ layout. This script bridges
the two: it collects every sequence's dense/mesh_poisson.ply into one flat
folder, named <date>_<seq_id>_mesh_poisson.ply, which is exactly the naming
labelCloud will then echo back in labels_root/<date>_<seq_id>_mesh_poisson.json
(what 05_build_dataset.py expects).
"""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
                    help="Root containing <date>/<seq_id>/dense/mesh_poisson.ply (step 3 output)")
    p.add_argument("--pointcloud_folder", type=str, required=True,
                    help="Shared folder to collect meshes into (point labelCloud's "
                         "config.ini pointcloud_folder here)")
    p.add_argument("--symlink", action="store_true",
                    help="Symlink instead of copy (saves disk space for large meshes)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.pointcloud_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_dirs = sorted(d for d in data_root.glob("*/*") if d.is_dir())
    n_gathered = 0
    for seq_dir in seq_dirs:
        mesh_path = seq_dir / "dense" / "mesh_poisson.ply"
        if not mesh_path.exists():
            continue
        date, seq_id = seq_dir.parent.name, seq_dir.name
        dest = out_dir / f"{date}_{seq_id}_mesh_poisson.ply"
        if args.symlink:
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            dest.symlink_to(mesh_path.resolve())
        else:
            shutil.copy2(mesh_path, dest)
        n_gathered += 1

    print(f"Gathered {n_gathered} mesh(es) into {out_dir}")
    print("Now open labelCloud with this as pointcloud_folder and annotate each one; "
          "point label_folder at your <labels_root> for step 5.")


if __name__ == "__main__":
    main()

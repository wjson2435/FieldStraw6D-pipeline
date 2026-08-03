"""Minimal readers for COLMAP binary model files (cameras.bin, images.bin).

Implements just enough of the COLMAP binary format to read camera intrinsics
and per-image poses -- no COLMAP/pycolmap installation required downstream.
"""
import struct
import numpy as np

CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12, 11: 5,
}


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path):
    """Returns {image_id: {"qvec": (4,), "tvec": (3,), "camera_id": int, "name": str}}."""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            c = _read_next_bytes(fid, 1, "c")[0]
            while c != b"\x00":
                name += c.decode("utf-8")
                c = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2d)  # skip (x, y, point3D_id) triplets
            images[image_id] = {"qvec": qvec, "tvec": tvec, "camera_id": camera_id, "name": name}
    return images


def read_cameras_binary(path):
    """Returns {camera_id: {"model_id": int, "width": int, "height": int, "params": tuple}}."""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            n = CAMERA_MODEL_NUM_PARAMS.get(model_id, 4)
            params = _read_next_bytes(fid, 8 * n, "d" * n)
            cameras[camera_id] = {
                "model_id": model_id, "width": width, "height": height, "params": params,
            }
    return cameras


def qvec2rotmat(qvec):
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix."""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
    ])

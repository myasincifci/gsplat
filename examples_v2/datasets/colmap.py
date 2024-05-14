import os
import sys
from typing import Any, Dict

import imageio.v2 as imageio
import numpy as np
import torch
from pycolmap import SceneManager

from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."

        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        manager.load_points3D()

        # Assume shared intrinsics between all cameras.
        cam = manager.cameras[1]
        fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        K[:2, :] /= factor

        # Extract extrinsic matrices in world-to-camera format.
        imdata = manager.images
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            rot = im.R()
            trans = im.tvec.reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)
        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Get distortion parameters.
        type_ = cam.camera_type

        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"

        elif type_ == 1 or type_ == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"

        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = np.array([cam.k1], dtype=np.float32)
            camtype = "perspective"

        elif type_ == 3 or type_ == "RADIAL":
            params = np.array([cam.k1, cam.k2], dtype=np.float32)
            camtype = "perspective"

        elif type_ == 4 or type_ == "OPENCV":
            params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            camtype = "perspective"

        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
            camtype = "fisheye"

        assert (
            camtype == "perspective"
        ), f"Only support perspective camera model, got {type_}"

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(os.listdir(colmap_image_dir))
        image_files = sorted(os.listdir(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3D points and {image_name -> [point_idx]}
        points = manager.points3D.astype(np.float32)
        points_err = manager.point3D_errors.astype(np.float32)
        points_rgb = manager.point3D_colors.astype(np.uint8)
        point_indices = dict()

        image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        for point_id, data in manager.point3D_id_to_images.items():
            for image_id, _ in data:
                image_name = image_id_to_name[image_id]
                point_idx = manager.point3D_id_to_point3D_idx[point_id]
                point_indices.setdefault(image_name, []).append(point_idx)
        point_indices = {
            k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        image = imageio.imread(image_paths[0])
        height, width = image.shape[:2]

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.K = K  # np.ndarray, (3, 3)
        self.params = params  # np.ndarray, (K,)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)
        self.height = height
        self.width = width

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)


class Dataset(Parser):
    """A simple dataset class."""

    def __init__(self, split: str = "train", **kwargs):
        super().__init__(**kwargs)
        self.split = split
        indices = np.arange(len(self.image_names))
        if split == "train":
            self.indices = indices[indices % self.test_every != 0]
        else:
            self.indices = indices[indices % self.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item % len(self.indices)]
        return {
            "K": torch.from_numpy(self.K.copy()).float(),
            "camtoworld": torch.from_numpy(self.camtoworlds[index]).float(),
            "image": torch.from_numpy(
                imageio.imread(self.image_paths[index])[..., :3]
            ).float(),
        }


if __name__ == "__main__":
    # Test / Usage
    data_dir = "data/360_v2/garden"  # 185 images
    parser = Parser(data_dir=data_dir, factor=4, normalize=True, test_every=8)
    print(f"{len(parser.image_names)} images for data_dir: {data_dir}")
import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim

from utils import readCamerasFromTransforms
from random import randint
import copy

class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        cameras,
        num_points: int = 2000,
    ):
        self.cameras = cameras

        self.device = torch.device("cuda:0")
        # self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        self.yxz_grad_accum = None
        self.grad_denom = 0

        # fov_x = fov_x # math.pi / 2.0
        # self.H, self.W = H, W # gt_image.shape[0], gt_image.shape[1]
        # self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        # self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        # self.viewmat = viewmat

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2.0
        be = 0.000001

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = be * (torch.rand(self.num_points, 3, device=self.device))
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.quats = self.quats.to(device=self.device)
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        # self.viewmat = torch.tensor(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0.0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 9.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device=self.device,
        # )
        # self.viewmat = self.viewmat.to(self.device)
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        # self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        B_SIZE: int = 14,
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.L1Loss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16
        for iter in range(iterations):
            idx = randint(1, len(self.cameras)) - 1
            camera = copy.copy(self.cameras[idx])

            viewmat = torch.from_numpy(camera.w2c).to(dtype=torch.float).to(self.device)
            fov_x = camera.FovX
            W = camera.width
            H = camera.height
            gt_image = image_path_to_tensor(camera.image_path).to(self.device)
            focal = 0.5 * float(W) / math.tan(0.5 * fov_x)

            start = time.time()
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                num_tiles_hit,
                cov3d,
            ) = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                viewmat,
                focal,
                focal,
                W / 2,
                H / 2,
                H,
                W,
                B_SIZE,
            )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                H,
                W,
                B_SIZE,
                self.background,
            )[..., :3]
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            # with torch.no_grad():
            #     # Densify
            #     if self.yxz_grad_accum:
            #         self.xyz_grad_accum += torch.norm(xys.grad[:2], dim=-1, keepdim=True)
            #     else:
            #         self.xyz_grad_accum = torch.norm(xys.grad[:2], dim=-1, keepdim=True)[:]
            #     self.grad_denom += 1

            #     if iter % 100 == 0:
            #         grads = self.xyz_grad_accum / self.grad_denom
            #         grads[grads.isnan()] = 0.0

                    

            #         torch.cuda.empty_cache()

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/training.gif",
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
        print(
            f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        )
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor


def main(
    height: int = 256,
    width: int = 256,
    num_points: int = 100000,
    save_imgs: bool = True,
    img_path: Optional[Path] = None,
    iterations: int = 1000,
    lr: float = 0.01,
) -> None:
    cameras = readCamerasFromTransforms(
        # './examples/nerf_example_data/nerf_synthetic/lego',
        './nerf_example_data/nerf_synthetic/lego',
        'transforms_train.json',
        False
    )

    trainer = SimpleTrainer(cameras, num_points=num_points)
    trainer.train(
        iterations=iterations,
        lr=lr,
        save_imgs=save_imgs,
    )


if __name__ == "__main__":
    tyro.cli(main)

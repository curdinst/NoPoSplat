import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetRE10kCfgWrapper:
    re10k: DatasetRE10kCfg


@dataclass
class DatasetDL3DVCfgWrapper:
    dl3dv: DatasetRE10kCfg


@dataclass
class DatasetScannetppCfgWrapper:
    scannetpp: DatasetRE10kCfg


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        # if self.stage in ("train", "val"):
        #     self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            # item = [x for x in chunk if x["key"] == "1214f2a11a9fc1ed"]
            item = [x for x in chunk if x["key"] == "c48f19e2ffa52523"]
            print("len(item)", len(item))
            # assert len(item) == 1
            if len(item) != 1:
                continue
            chunk = item * len(chunk)
            # print("chunk before shuffle", [example["key"] for example in chunk])
            # if self.stage in ("train", "val"):
            #     chunk = self.shuffle(chunk)

            # print("chunk after shuffle", [example["key"] for example in chunk])
            # example = chunk[0]
            # for i in range(2):a
            
            # for example in chunk:
            example = chunk[0]
            num_imgs = len(example["images"])
            # print("Num images", num_imgs)
            # img1, img2, img3 = 180, 210, 240
            # print("Input Sequence: ", img1, img2, img3)
            # target1, target2 = int((img2-img1)/2+img1), int((img3-img2)/2+img2)
            # print("Target frames: ", target1, target2)
            # context_indices_list = [torch.tensor([img1, img2]),torch.tensor([img2, img3])]
            # target_indices_list = [torch.tensor([0,target1,target2]),torch.tensor([target1,target2,num_imgs-1])]

            start_idx = 0
            end_idx = 40
            step = 20
            last_idx = start_idx
            last_last_idx = start_idx
            context_indices_list = []
            target_indices_list = []
            for i in range(start_idx, end_idx, step):
                if i == start_idx: continue
                context_indices_list.append(torch.tensor([last_idx, i]))
                target_indices_list.append(torch.tensor([max(last_last_idx, 0), i, i+step]))
                last_last_idx = last_idx
                last_idx = i
            print("Context pairs list: ", context_indices_list)

            for i in range(len(context_indices_list)):
                print("inputs: ", context_indices_list[i])
                print("targets", target_indices_list[i])
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices, overlap = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    context_indices = context_indices_list[i]
                    target_indices = target_indices_list[i]
                    # print("overlap", overlap)
                    # print("context_indices", context_indices)
                    # print("target_indices", target_indices)
                    # print("key", example["key"])
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    print("Skipped", example["key"])
                    continue
                scene = example["key"]+"_"+str(context_indices[0].item())+"_"+str(context_indices[1].item())
                print("test, ", scene)
                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                try:
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]
                    context_images = self.convert_images(context_images)
                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]
                    target_images = self.convert_images(target_images)
                except IndexError:
                    print("IndexError")
                    continue
                except OSError:
                    print(f"Skipped bad example {example['key']}.")  # DL3DV-Full have some bad images
                    continue

                # Skip the example if the images don't have the right shape.
                context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
                target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                if self.cfg.make_baseline_1:
                    print("resize to make_baseline_1")
                    a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    scale = (a - b).norm()
                    if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                        print(
                            f"Skipped {scene} because of baseline out of range: "
                            f"{scale:.6f}"
                        )
                        continue
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                if self.cfg.relative_pose:
                    extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

                example_out = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / scale,
                        "far": self.get_bound("far", len(context_indices)) / scale,
                        "index": context_indices,
                        "overlap": overlap,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / scale,
                        "far": self.get_bound("far", len(target_indices)) / scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }
                if self.stage == "train" and self.cfg.augment:
                    example_out = apply_augmentation_shim(example_out)
                yield apply_crop_shim(example_out, tuple(self.cfg.input_image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())

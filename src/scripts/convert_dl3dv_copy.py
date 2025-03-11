import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
from einops import rearrange, repeat
import cv2

INPUT_IMAGE_DIR = Path("/home/curdin/datasets/DL3DV_subset/4K")
METADATA_DIR = Path("/home/curdin/datasets/DL3DV_subset")
OUTPUT_DIR = Path("/home/curdin/datasets/DL3DV_subset_torch/test")


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]

def opengl_c2w_to_opencv_w2c(c2w: np.ndarray) -> np.ndarray:
    c2w = c2w.copy()
    c2w[2, :] *= -1
    c2w = c2w[np.array([1, 0, 2, 3]), :]
    c2w[0:3, 1:3] *= -1
    w2c_opencv = np.linalg.inv(c2w)
    return w2c_opencv

def load_metadata(file_path: Path) -> Metadata:
    with open(file_path, 'r') as file:
        data = json.load(file)

    url = ""

    timestamps = []
    cameras = []

    # FIXME: igore k1, k2, p1, p2, is this proper?
    w = data['w']
    h = data['h']
    print("w, h:", w, h)
    intrinsic = [data['fl_x'] / w, data['fl_y'] / h, data['cx'] / w, data['cy'] / h, 0.0, 0.0]
    intrinsic = np.array(intrinsic, dtype=np.float32)

    for frame in data['frames']:
        # extract number from string like "images/frame_00002.png"
        frame_id = int(frame['file_path'].split('_')[-1].split('.')[0])
        if frame_id not in frame_numbers:
            continue
        timestamps.append(frame_id)
        extrinsic = frame['transform_matrix']
        extrinsic = np.array(extrinsic, dtype=np.float32)
        # print(extrinsic)
        w2c = opengl_c2w_to_opencv_w2c(extrinsic)
        w2c = w2c[:3, :]
        w2c = w2c.flatten()
        camera = np.concatenate([intrinsic, w2c])
        cameras.append(camera)

    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)
    print(cameras)
    return {
        "url": url,
        "timestamps": timestamps,
        "cameras": cameras,
    }

def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))

def load_images(image_paths: list[Path]) -> list[UInt8[Tensor, "..."]]:
    # return [load_raw(path) for path in image_paths]
    return {path.stem: load_raw(path) for path in image_paths}

def save_example(example: Example, output_path: Path):
    torch.save(example, output_path / f"test_file.torch")
    # torch.save(example, output_path)

def convert_poses(
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

if __name__ == "__main__":
    # Paths to the two PNG images

    frame_numbers = [10, 12, 13, 14, 15]
    image_paths = [
        INPUT_IMAGE_DIR.joinpath(Path("frame_000"+str(frame_numbers[0])+".png")),
        INPUT_IMAGE_DIR.joinpath(Path("frame_000"+str(frame_numbers[1])+".png")),
        INPUT_IMAGE_DIR.joinpath(Path("frame_000"+str(frame_numbers[2])+".png")),
        INPUT_IMAGE_DIR.joinpath(Path("frame_000"+str(frame_numbers[3])+".png")),
        INPUT_IMAGE_DIR.joinpath(Path("frame_000"+str(frame_numbers[4])+".png"))
    ]

    # Load the images
    images = load_images(image_paths)
    example = load_metadata(METADATA_DIR.joinpath("transforms.json"))
    print("example:", example)
    # Merge the images into the example.
    # from int to "frame_00001" format
    image_names = [f"frame_{timestamp.item():0>5}" for timestamp in example["timestamps"]]
    print("image_names:", image_names)
    try:
        example["images"] = [
            images[image_name] for image_name in image_names
        ]
    except KeyError:
        print(f"Skipping key because of missing images.")
    assert len(example["images"]) == len(example["timestamps"]), f"len(example['images'])={len(example['images'])}, len(example['timestamps'])={len(example['timestamps'])}"

    print("cameras\n" ,convert_poses(example["cameras"]))
    # Add the key to the example.
    example["key"] = "5aca87f95a9412c6"

    # print(f"    Added {key} to chunk ({num_bytes / 1e6:.2f} MB).")
    # chunk.append(example)

    # Output path for the torch file
    output_path = OUTPUT_DIR

    example_list = [example]
    # print(example)
    # Save the example
    save_example(example_list, output_path)

    print(f"Saved example to {output_path}")

    print("-----------------")

    # original_torch_file_path = Path("/home/curdin/noposplat/NoPoSplat/datasets/re10k_orig/test/")
    test_file = torch.load(output_path / "test_file.torch")
    # test_file = torch.load(original_torch_file_path / "000000.torch")
    print(len(test_file))
    # print(test_file[0]["timestamps"])
    print(test_file[0].keys())
    print(test_file[0]["key"])


    stamps = test_file[0]["timestamps"]
    images = test_file[0]["images"]
    print("len images", len(images))
    import matplotlib.pyplot as plt

    # Convert the first image tensor to a numpy array
    image_np = images[0].numpy()
    print("image_np shape", image_np.shape) 
    # Reshape the image tensor to a 2D array (height, width, channels)
    # image_np = image_np.reshape((74,547))
    # Convert the numpy array to a cv2 image
    # image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # # Display the image using matplotlib
    # plt.imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()


    # for i in range(20239):
    #     if i != 0:
    #         if 20239/i %1 == 0:
    #             print(20239/i)
    # Subtract the last timestamp from each timestamp in the example
    # last_stamp = -1
    # for stamp in stamps:
    #     if last_stamp == -1:
    #         last_stamp = stamp
    #     else:
    #         diff = stamp - last_stamp
    #         print(f"diff: {diff}")
    #         last_stamp = stamp


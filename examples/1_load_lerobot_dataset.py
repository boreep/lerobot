"""
This script demonstrates the use of `LeRobotDataset` class for handling and processing robotic datasets from Hugging Face.
It illustrates how to load datasets, manipulate them, and apply transformations suitable for machine learning tasks in PyTorch.

Features included in this script:
- Loading a dataset and accessing its properties.
- Filtering data by episode number.
- Converting tensor data for visualization.
- Saving video files from dataset frames.
- Using advanced dataset features like timestamp-based frame selection.
- Demonstrating compatibility with PyTorch DataLoader for batch processing.

The script ends with examples of how to batch process data using PyTorch's DataLoader.
"""
"""
该脚本演示了如何使用“LeRobotDataset”类来处理和处理来自 Hugging Face 的机器人数据集。
它说明了如何加载数据集、操作数据集以及应用适合 PyTorch 中的机器学习任务的转换。
该类会自动从Hugging Face中心下载数据

该脚本包含的功能：
-加载数据集并访问其属性。
-按剧集编号过滤数据。
-转换张量数据以进行可视化。
-从数据集帧保存视频文件。
-使用高级数据集功能，例如基于时间戳的帧选择。
-演示与 PyTorch DataLoader 的批处理兼容性。

该脚本最后提供了如何使用 PyTorch 的 DataLoader 批量处理数据的示例
"""

from pathlib import Path
from pprint import pprint

import imageio
import torch

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

print("List of available datasets:")
pprint(lerobot.available_datasets)

# Let's take one for this example
repo_id = "lerobot/pusht"

# You can easily load a dataset from a Hugging Face repository
dataset = LeRobotDataset(repo_id)

#非在线方式
"""
# 定义本地数据目录
LOCAL_DATA_DIR = "/path/to/local/dataset"
DATA_DIR = Path(LOCAL_DATA_DIR)

print("List of available datasets:")
pprint(lerobot.available_datasets)

# Let's take one for this example
repo_id = "lerobot/pusht"

# 通过将本地数据目录传递给构造函数来加载数据集
dataset = LeRobotDataset(repo_id, root=DATA_DIR)
"""

# LeRobotDataset is actually a thin wrapper around an underlying Hugging Face dataset
# (see https://huggingface.co/docs/datasets/index for more information).
print(dataset)
print(dataset.hf_dataset)

# And provides additional utilities for robotics and compatibility with Pytorch
print(f"\naverage number of frames per episode: {dataset.num_samples / dataset.num_episodes:.3f}")
print(f"frames per second used during data collection: {dataset.fps=}")
print(f"keys to access images from cameras: {dataset.camera_keys=}\n")

# Access frame indexes associated to first episode
episode_index = 0
from_idx = dataset.episode_data_index["from"][episode_index].item()
to_idx = dataset.episode_data_index["to"][episode_index].item() #不包括这个值本身，即需要-1

# LeRobot datasets actually subclass PyTorch datasets so you can do everything you know and love from working
# with the latter, like iterating through the dataset. 这里我们抓取所有帧.
frames = [dataset[idx]["observation.image"] for idx in range(from_idx, to_idx)]

# Video frames are now float32 in range [0,1] channel first (c,h,w) to follow pytorch convention. To visualize
# them, we convert to uint8 in range [0,255]，
# 转换为通道优先的uint8张量，范围为[0,255]，并改变维度顺序
frames = [(frame * 255).type(torch.uint8) for frame in frames]
# and to channel last (h,w,c).
frames = [frame.permute((1, 2, 0)).numpy() for frame in frames]

# Finally, we save the frames to a mp4 video for visualization.保存为MP4文件
Path("outputs/examples/1_load_lerobot_dataset").mkdir(parents=True, exist_ok=True)
imageio.mimsave("outputs/examples/1_load_lerobot_dataset/episode_0.mp4", frames, fps=dataset.fps)

# For many machine learning applications we need to load the history of past observations or trajectories of
# future actions. Our datasets can load previous and future frames for each key/modality, using timestamps
# differences with the current loaded frame. For instance:
#读取特殊时间帧下的数据
delta_timestamps = {
    # loads 4 images: 1 second before current frame, 500 ms before, 200 ms before, and current frame
    "observation.image": [-1, -0.5, -0.20, 0],
    # loads 8 state vectors: 1.5 seconds before, 1 second before, ... 20 ms, 10 ms, and current frame
    "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, -0.02, -0.01, 0],
    # loads 64 action vectors: current frame, 1 frame in the future, 2 frames, ... 63 frames in the future
    "action": [t / dataset.fps for t in range(64)],
}
dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
print(f"\n{dataset[0]['observation.image'].shape=}")  # (4,c,h,w)
print(f"{dataset[0]['observation.state'].shape=}")  # (8,c)
print(f"{dataset[0]['action'].shape=}\n")  # (64,c)

# Finally, our datasets are fully compatible with PyTorch dataloaders and samplers because they are just
# PyTorch datasets.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=32,
    shuffle=True,
)
for batch in dataloader:
    print(f"{batch['observation.image'].shape=}")  # (32,4,c,h,w)
    print(f"{batch['observation.state'].shape=}")  # (32,8,c)
    print(f"{batch['action'].shape=}")  # (32,64,c)
    break

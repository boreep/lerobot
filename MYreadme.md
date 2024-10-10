
## 文件结构

```
.
├── examples             # 包括一些示例文件，从这里开始学习。
|   └── advanced         # 包括更多示例文件，建议掌握基础后使用。
├── lerobot
|   ├── configs          # 包括可以在命令行覆盖配置的配置文件
|   |   ├── default.yaml   # 默认选择，加载pusht环境和diffusion policy
|   |   ├── env            # 不同的仿真环境和对应的数据集
|   |   └── policy         # 不同策略的配置文件，包括：ACT,DP,TDMPC
|   ├── common           # 类和实用工具集合
|   |   ├── datasets       # 不同的人类演示：aloha, pusht, xarm
|   |   ├── envs           # 不同的放则很难环境: aloha, pusht, xarm
|   |   ├── policies       # 不同策略: act, diffusion, tdmpc
|   |   ├── robot_devices  # 不同真实设备: dynamixel motors, opencv cameras, koch robots
|   |   └── utils          # 不同实用工具: logger, data_utils, visualization
|   └── scripts          # 包含通过命令行执行的函数
|       ├── eval.py                 # 加载训练后策略并在环境中评估其性能
|       ├── train.py                # 训练策略
|       ├── control_robot.py        # 遥操作真实机器人、记录数据、部署策略
|       ├── push_dataset_to_hub.py  # 将你的数据集格式转换为LeroBot的格式并上传到
|       └── visualize_dataset.py    # 加载数据集并渲染可视化结果
├── outputs               # 包括脚本执行输出结果：logs，video，检查点等
└── tests                 # pytest实用工具用于持续集成
```

## 数据集结构
数据集结构以 ` lerobot/aloha_static_coffee `实例化的典型`LeRobotDataset`为例，其他数据集也大差不差。
```
dataset attributes:
  ├ hf_dataset包含了从 Hugging Face Hub 加载的数据集，格式通常是 Arrow/Parquet 文件。 具体格式请参考:
  │  ├ observation.images.cam_high (VideoFrame):
  │  │   VideoFrame = {'path': mp4视频路径, 'timestamp' (float32): 视频时间戳}
  │  ├ observation.state (list of float32): 手臂关节位置等状态信息
  │  ... (更多观察量)
  │  ├ action (list of float32):目标关节位置等动作信息
  │  ├ episode_index (int64): 样本所属的episode_index（剧集索引）
  │  ├ frame_index (int64): 样本在剧集中的帧索引。
  │  ├ timestamp (float32): 样本在剧集中的时间戳。
  │  ├ next.done (bool):指示是否为剧集的最后一帧。
  │  └ index (int64): 样本在整个数据集中的通用索引。
  ├ episode_data_index: 包含了每个剧集的起始和结束索引。
  │  ├ from (1D int64 tensor):每个剧集的第一帧索引。— shape (num episodes,)从0开始
  │  └ to: (1D int64 tensor): 每个剧集的最后一帧索引（不包括这一帧本身）
  ├ stats: 包含了数据集中每个特征的最大值、平均值、最小值和标准差等统计信息。
  │  ├ observation.images.cam_high: {'max': tensor with same number of dimensions (e.g. `(c, 1, 1)` for images, `(c,)` for states), etc.}
  │  ...
  ├ info: 包含了关于数据集的元数据信息。
  │  ├ codebase_version (str): 创建数据集时使用的代码版本。
  │  ├ fps (float): 数据集中帧的每秒帧数。
  │  ├ video (bool): 表示帧是否存储为 MP4 视频文件。
  │  └ encoding (dict): 如果存储为视频，则记录了编码选项。
  ├ videos_dir (Path): 存储 MP4 视频或 PNG 图像的目录。
  └ camera_keys (list of string): 访问相机特征的键列表。 (e.g. `["observation.images.cam_high", ...]`)
```
`LeRobotDataset`的每个部分都使用几种广泛使用的文件格式进行序列化，即：
- hf_dataset使用 Hugging Face 数据集库序列化为 parquet 进行存储
- 视频以 MP4 格式存储以节省空间或 PNG 文件
- episode_data_index使用 `SafeTensor` 张量序列化格式保存
- stats使用 `SafeTensor`张量序列化格式保存
- info 使用 JSON 保存


## 加载数据集示例 `/examples/1_load_lerobot_dataset.py`
该脚本演示了如何使用 `LeRobotDataset`类来处理和处理来自 **Hugging Face** 的机器人数据集。它说明了如何加载数据集、操作数据集以及应用适合 PyTorch 中的机器学习任务的转换。

该脚本包含的功能：
- 加载数据集并访问其属性。__（数据集可以来自云端下载或者在本地加载）__
- 按数剧集编号过滤数据。
- 转换张量数据以进行可视化。
- 从数据集帧保存视频文件。
- 使用高级数据集功能，例如基于时间戳的帧选择。
- 演示与 PyTorch DataLoader 的批处理兼容性。

该脚本最后提供了如何使用 PyTorch 的 DataLoader 批量处理数据的示例



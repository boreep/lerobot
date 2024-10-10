
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
## 运行LeroBot
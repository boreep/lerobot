"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""
"""
此脚本演示了如何评估来自 HuggingFace Hub 或本地的预训练策略
。在后一种情况下，您可能需要先运行 Examples/3_train_policy.py来训练策略并得到输出目录。
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
from huggingface_hub import snapshot_download

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

# Create a directory to store the video of the evaluation
# 目录用于存放评估视频
output_directory = Path("outputs/eval/example_pusht_diffusion")
output_directory.mkdir(parents=True, exist_ok=True)

# Download the diffusion policy for pusht environment
# 从 HuggingFace Hub 下载 diffusion 策略
pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))

# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# 或是取消注释内容从本地的 outputs/train 文件夹中评估策略
# pretrained_policy_path = Path("outputs/train/example_pusht_diffusion")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Device set to:", device)
else:
    device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {device}. Inference will be slower than on GPU.")
    # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
    policy.diffusion.num_inference_steps = 10

policy.to(device)

#  初始化评估环境以渲染两种观察类型（动作和图像）
# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "gym_pusht/PushT-v0",
    obs_type="pixels_agent_pos",
    max_episode_steps=300,
)

# Reset the policy and environmens to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
#准备收集所有奖励和整个回合的帧。
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
while not done:
    # Prepare observation for the policy running in Pytorch
    #将观察转换为PyTorch可运行的格式
    state = torch.from_numpy(numpy_observation["agent_pos"])
    image = torch.from_numpy(numpy_observation["pixels"])

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    #将图像从通道优先的[0,255]转换为通道后[0,1]的float32
    state = state.to(torch.float32)
    image = image.to(torch.float32) / 255
    image = image.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image = image.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    #这里 unsqueeze(0) 方法被用来增加一个额外的维度（batch dimension）到张量前面，从而批处理，为了适应已经为批量输入设计的模型
    state = state.unsqueeze(0)
    image = image.unsqueeze(0)

    # Create the policy input dictionary

    observation = {
        "observation.state": state,
        "observation.image": image,
    }

    # Predict the next action with respect to the current observation
    #预测下一个动作，以当前观察为依据

    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    #运行动作，将一个包含动作的 PyTorch 张量转换为 NumPy 数组，
    #这通常是在你需要将张量的数据用于非 PyTorch 的环境中使用时进行的操作
    numpy_action = action.squeeze(0).to("cpu").numpy()

    # Step through the environment and receive a new observation
    #在环境中运行一步，并接收一个新的观察
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    #达到成功状态（即terminated为 True）时，部署被认为已完成，
    #或者达到最大迭代次数（即 truncated 为 True）
    done = terminated | truncated | done
    step += 1

if terminated:
    print("Success!")
else:
    print("Failure!")

# Get the speed of environment (i.e. its number of frames per second).
#获取环境的速度（即每秒帧数）。
fps = env.metadata["render_fps"]

# Encode all frames into a mp4 video.
#将所有帧编码为 mp4 视频。
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")

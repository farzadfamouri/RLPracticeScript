# ============================================================
# A3C-style Atari RL with Gymnasium + ALE-py (Farama)
# Environment: ALE/KungFuMaster-v5
# ============================================================

# pip install:
# gymnasium[atari,accept-rom-license] ale-py torch torchvision numpy opencv-python tqdm

import math
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
import gymnasium as gym
import ale_py
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
import tqdm

# ============================================================
# 0. Register Atari environments
# ============================================================

gym.register_envs(ale_py)

# Quick sanity check
def breakout_sanity_check():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.close()
    print("Breakout test successful ✅")

breakout_sanity_check()

# ============================================================
# 1. Neural Network (Actor-Critic)
# ============================================================

class Network(nn.Module):
    def __init__(self, action_size):
        super(Network, self).__init__()
        # Input: (batch, 4, 42, 42)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2)   # -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)  # -> (32, 9, 9)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)  # -> (32, 4, 4)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2a = nn.Linear(128, action_size)  # actor (policy logits)
        self.fc2s = nn.Linear(128, 1)           # critic (state value)

    def forward(self, state):
        # state: (batch, 4, 42, 42) or (4,42,42)
        if state.dim() == 3:
            state = state.unsqueeze(0)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        action_logits = self.fc2a(x)
        state_value = self.fc2s(x).squeeze(-1)   # (batch,)
        return action_logits, state_value

# ============================================================
# 2. Preprocessing Wrapper
# ============================================================

class PreprocessAtari(ObservationWrapper):
    """
    - Resizes to (height, width)
    - Converts to grayscale (if color=False)
    - Normalizes to [0,1]
    - Stacks last n_frames along channel dimension (PyTorch order)
    """

    def __init__(
        self,
        env,
        height=42,
        width=42,
        crop=lambda img: img,
        dim_order='pytorch',
        color=False,
        n_frames=4
    ):
        super().__init__(env)
        self.height = height
        self.width = width
        self.crop = crop
        self.dim_order = dim_order
        self.color = color
        self.n_frames = n_frames

        if self.color:
            n_channels = 3 * n_frames
        else:
            n_channels = n_frames

        if dim_order == 'pytorch':
            obs_shape = (n_channels, height, width)
        else:
            obs_shape = (height, width, n_channels)

        self.observation_space = Box(
            low=0.0, high=1.0,
            shape=obs_shape,
            dtype=np.float32
        )
        self.frames = np.zeros(obs_shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames[...] = 0.0
        self.update_buffer(obs)
        return self.frames, info

    def observation(self, obs):
        # obs from env: RGB (H,W,3), uint8
        img = self.crop(obs)
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)

        if not self.color:
            # Input is RGB; convert to grayscale correctly
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = img.astype(np.float32) / 255.0  # (H, W)
            if self.dim_order == 'pytorch':
                # Roll along channel axis (0)
                self.frames = np.roll(self.frames, -1, axis=0)
                self.frames[-1, :, :] = img
            else:
                self.frames = np.roll(self.frames, -1, axis=2)
                self.frames[:, :, -1] = img
        else:
            img = img.astype(np.float32) / 255.0  # (H, W, 3)
            if self.dim_order == 'pytorch':
                img = np.transpose(img, (2, 0, 1))  # (3, H, W)
                self.frames = np.roll(self.frames, -3, axis=0)
                self.frames[-3:, :, :] = img
            else:
                self.frames = np.roll(self.frames, -3, axis=2)
                self.frames[:, :, -3:] = img

        return self.frames

    def update_buffer(self, obs):
        # Fill buffer with initial frame stack
        _ = self.observation(obs)

# ============================================================
# 3. Environment factory
# ============================================================

def make_env():
    env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
    env = PreprocessAtari(
        env,
        height=42,
        width=42,
        crop=lambda img: img,
        dim_order='pytorch',
        color=False,
        n_frames=4
    )
    return env

# Single env to inspect
env = make_env()
state_shape = env.observation_space.shape
number_actions = env.action_space.n

print("State shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.unwrapped.get_action_meanings())

# Quick random rollout test
obs, info = env.reset()
total_reward = 0.0
for _ in range(20):
    a = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(a)
    total_reward += reward
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Random rollout done, total reward:", total_reward)

# ============================================================
# 4. A3C-style Agent (single-process variant)
# ============================================================

learning_rate = 1e-4
discount_factor = 0.99
number_environments = 10

class Agent:
    def __init__(self, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.network = Network(action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

    def act(self, state_batch):
        # state_batch: (batch, C, H, W) or (C,H,W)
        if isinstance(state_batch, np.ndarray):
            state = torch.tensor(state_batch, dtype=torch.float32, device=self.device)
        else:
            state = state_batch.to(self.device)

        if state.dim() == 3:
            state = state.unsqueeze(0)

        logits, _ = self.network(state)
        probs = F.softmax(logits, dim=-1)
        probs_np = probs.detach().cpu().numpy()

        actions = np.array([np.random.choice(len(p), p=p) for p in probs_np])
        return actions

    def step(self, state, action, reward, next_state, done):
        # All are batched
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

        logits, state_value = self.network(state)
        with torch.no_grad():
            _, next_state_value = self.network(next_state)

        target_state_value = reward + discount_factor * next_state_value * (1.0 - done)
        advantage = target_state_value - state_value

        probs = F.softmax(logits, dim=-1)
        logprobs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * logprobs).sum(dim=-1)

        batch_idx = torch.arange(state.shape[0], device=self.device)
        logp_actions = logprobs[batch_idx, action]

        actor_loss = -(logp_actions * advantage.detach()).mean() - 0.001 * entropy.mean()
        critic_loss = F.mse_loss(state_value, target_state_value.detach())
        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ============================================================
# 5. Evaluation function
# ============================================================

def evaluate(agent, env, n_episodes=1):
    rewards = []
    for _ in range(n_episodes):
        state, info = env.reset()
        total_reward = 0.0
        while True:
            action = agent.act(state)[0]
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            state = next_state
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return rewards

# ============================================================
# 6. Vectorized environment manager
# ============================================================

class EnvBatch:
    def __init__(self, n_envs=10):
        self.envs = [make_env() for _ in range(n_envs)]

    def reset(self):
        states = []
        for env in self.envs:
            s, _ = env.reset()
            states.append(s)
        return np.stack(states, axis=0)

    def step(self, actions):
        next_states = []
        rewards = []
        dones = []
        infos = []

        for env, a in zip(self.envs, actions):
            obs, r, terminated, truncated, info = env.step(int(a))
            done = terminated or truncated
            if done:
                obs, _ = env.reset()
            next_states.append(obs)
            rewards.append(r)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(next_states, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos
        )

# ============================================================
# 7. Training Loop (A3C-style, single process)
# ============================================================

agent = Agent(number_actions)
env_batch = EnvBatch(number_environments)
batch_states = env_batch.reset()

with tqdm.trange(0, 3001) as progress_bar:
    for i in progress_bar:
        batch_actions = agent.act(batch_states)
        batch_next_states, batch_rewards, batch_dones, _ = env_batch.step(batch_actions)

        # Scale rewards a bit to stabilize (from your original code)
        batch_rewards_scaled = batch_rewards * 0.01

        agent.step(batch_states, batch_actions, batch_rewards_scaled, batch_next_states, batch_dones)
        batch_states = batch_next_states

        if i % 1000 == 0:
            avg_reward = np.mean(evaluate(agent, make_env(), n_episodes=5))
            print(f"\nStep {i} - Average eval reward over 5 episodes: {avg_reward}")
"""
# ============================================================
# 8. Visualization (recording and showing agent performance)
# ============================================================

import imageio
import base64
import io
from IPython.display import HTML, display


def record_agent_video(agent, filename="agent_play.mp4", max_steps=1000):
    
    env = gym.make("ALE/KungFuMaster-v5", render_mode="rgb_array")
    env = PreprocessAtari(env, height=84, width=84, n_frames=4, color=False)

    state, info = env.reset()
    frames = []
    total_reward = 0.0
    steps = 0

    while True:
        frame = env.render()
        frames.append(frame)

        action = agent.act(state)[0]
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

        if terminated or truncated or steps >= max_steps:
            break

    env.close()
    print(f"Total reward in recorded episode: {total_reward:.2f}")
    imageio.mimsave(filename, frames, fps=30)
    print(f"Video saved as {filename}")
    return filename


def show_recorded_video(filename="agent_play.mp4"):
    
    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    display(HTML(data=f'''
        <video alt="Agent gameplay" autoplay loop controls style="height:400px;">
            <source src="data:video/mp4;base64,{encoded.decode('ascii')}" type="video/mp4" />
        </video>'''))


# --- Example usage after training ---
video_path = record_agent_video(agent, filename="kungfu_agent.mp4", max_steps=1000)
show_recorded_video(video_path)


# Part 3 - Visualizing the results

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()

"""
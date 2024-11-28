import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import ReplayBuffer

from RL.CarlaEnv import CarlaEnv

data = np.load("expert_data.npz")
observations = data["observations"]
actions = data["actions"]

env = make_vec_env(CarlaEnv, n_envs=1)
model = PPO("CnnPolicy", env, verbose=1)

buffer = ReplayBuffer(len(observations), env.observation_space, env.action_space)
for obs, action in zip(observations, actions):
    buffer.add(obs, action, reward=0, done=False, infos={})

model.train_on_dataset(buffer, n_epochs=10)
model.save("ppo_pretrained")

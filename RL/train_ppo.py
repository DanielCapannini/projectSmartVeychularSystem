import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from CarlaEnv import CarlaEnv

env = CarlaEnv()

check_env(env)

model = PPO.load("ppo_pretrained", env=env)
model.learn(total_timesteps=100000)
model.save("ppo_carla_finetuned")

model = PPO.load("ppo_carla", env=env)

obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()

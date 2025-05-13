import gym

class GenesisWrapper(gym.Env):
    def __init__(self, task_id="WaterFranka-v0"):
        self.env = gym.make(task_id)
        self.observation_space = self.env.observation_space["image"]
        self.action_space      = self.env.action_space

    def reset(self):
        return self.env.reset()["image"]

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        return obs["image"], 0.0, done, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

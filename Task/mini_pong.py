import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

class MiniPongEnv(gym.Env):
    """A simplified Pong environment using the gym framework."""

    def __init__(self, level=5, size=5, normalise=True):
        super(MiniPongEnv, self).__init__()

        # Initialize parameters
        self.level = min(5, max(0, level))
        self.zmax = max(3, size) - 1
        self.size = 3 * max(3, size)
        self.xymax = self.size - 2
        self.scale = self.xymax if normalise else 1.0

        # Action space: 0 = no movement, 1 = left, 2 = right
        self.action_space = spaces.Discrete(3)

        # Observation space
        if self.level == 0:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(1, self.size, self.size), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.level,), dtype=np.float32)

        # Initial state
        self.reset()

    def reset(self):
        # Set initial positions and states
        x = random.randint(1, self.xymax)
        z = random.randint(0, self.zmax)
        self.s0 = (x, 2, z, 0, 0)
        self.s1 = (x, 2, z, 1, 1)
        self.p0 = np.zeros((self.size, self.size), dtype=int)
        self.p1 = np.zeros((self.size, self.size), dtype=int)
        return self._get_observation()

    def step(self, action):
        self.s0 = self.s1
        self.p0 = self.p1

        if self._is_terminal():
            return self._get_observation(), 0.0, True, {}

        x0, y0, z0, dx0, dy0 = self.s0

        # Paddle movement based on action
        if action == 1:
            z0 = max(0, z0 - 1)  # Move left
        elif action == 2:
            z0 = min(self.zmax, z0 + 1)  # Move right

        # Ball boundary and paddle collision logic
        if x0 == 1 and dx0 == -1:
            dx0 = 1
        if x0 == self.xymax and dx0 == 1:
            dx0 = -1
        if y0 == self.xymax and dy0 == 1:
            dy0 = -1
        if y0 == 2 and dy0 == -1 and x0 >= z0 * 3 and x0 < z0 * 3 + 3:
            dy0 = 1
            x0 = min(self.xymax, max(1, x0 + random.randint(-2, 2)))

        # Update ball position
        x1 = x0 + dx0
        y1 = y0 + dy0
        self.s1 = (x1, y1, z0, dx0, dy0)

        # Calculate reward and terminal condition
        reward = self._get_reward()
        done = self._is_terminal()

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        x1, y1, z1, dx1, dy1 = self.s1
        zx = z1 * 3 + 1
        dz = x1 - zx
        x0, y0, _, _, _ = self.s0
        dx = x1 - x0
        dy = y1 - y0

        if self.level == 0:
            self.p1 = self._to_pixels(self.s1, binary=True)
            return (self.p1 - 0.5 * self.p0).reshape(1, self.size, self.size)
        elif self.level == 1:
            return np.array([dz / self.scale])
        elif self.level == 2:
            return np.array([y1 / self.scale, dz / self.scale])
        elif self.level == 3:
            return np.array([y1 / self.scale, dx, dz / self.scale])
        elif self.level == 4:
            return np.array([x1 / self.scale, y1 / self.xymax, dx, dy])
        return np.array([x1 / self.xymax, y1 / self.xymax, dx, dy, dz / self.xymax])

    def _get_reward(self):
        x, y, z, dx, dy = self.s1
        if y == 2 and dy == -1 and x >= z * 3 and x < (z + 1) * 3:
            return self.xymax
        return 0.0

    def _is_terminal(self):
        _, y, _, _, _ = self.s1
        return y == 1

    def _to_pixels(self, state, binary=False):
        x, y, z, _, _ = state
        paddle_value = 1 if binary else -1
        image = np.zeros((self.size, self.size), dtype=int)
        image[self.size - 1, 0] = paddle_value
        image[self.size - 1, self.size - 1] = 1

        for i in range(3):
            image[0, z * 3 + i] = paddle_value
            image[y, x + i - 1] = 1
        image[y - 1, x] = 1
        image[y + 1, x] = 1

        return image

    def render(self, mode='human'):
        pix = self._to_pixels(self.s1)
        fig, ax = plt.subplots()
        ax.axis("off")
        plt.imshow(pix, origin='lower', cmap=plt.cm.coolwarm)
        plt.show()

# To test the environment using gym's built-in functions
if __name__ == "__main__":
    env = MiniPongEnv(level=1, size=5)
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}")

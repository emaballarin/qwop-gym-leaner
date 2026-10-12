# =============================================================================
# Copyright 2023 Simeon Manolov <s.manolloff@gmail.com>.
#           2025 Emanuele Ballarin <emanuele@ballarin.cc>
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import time

import gymnasium as gym
import numpy as np

from ..envs.v1.util.wsproto import WSProto

__all__: list[str] = ["VerboseWrapper"]


class VerboseWrapper(gym.Wrapper):
    """On each step, prints action along with some game stats"""

    def __init__(self, env):
        super().__init__(env=env)
        self.n_steps = 0
        self.total_reward = np.float32(0)
        self.last_distance = np.float32(0)
        self.last_time = np.float32(0)
        self.start_time = time.time()
        self.enabled = True

    def reset(self, *args, **kwargs):
        self.n_steps = 0
        self.total_reward = np.float32(0)
        self.last_distance = np.float32(0)
        self.last_time = np.float32(0)
        self.start_time = time.time()
        obs, info = self.env.reset(*args, **kwargs)
        self.last_distance = info["distance"]
        self.last_time = info["time"]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.enabled:
            return obs, reward, terminated, truncated, info

        self.n_steps += 1
        self.total_reward += reward

        flags = self.env.unwrapped.action_cmdflags[action]

        if terminated:
            keys = "XXXX"
        else:
            keys = ""
            keys += "Q" if flags & WSProto.CMD_K_Q else "."
            keys += "W" if flags & WSProto.CMD_K_W else "."
            keys += "O" if flags & WSProto.CMD_K_O else "."
            keys += "P" if flags & WSProto.CMD_K_P else "."

        ds = info["distance"] - self.last_distance
        dt = info["time"] - self.last_time
        v = ds / dt if dt != 0 else 0.0

        print(
            f"{self.n_steps:<5d} | {action!s:<4} | {keys:<4} | "
            f"{round(info['distance'], 1)!s:<6}m | {round(v, 1)!s:<6} m/s | "
            f"{round(reward, 2)!s:<6} | "
            f"{int(info['time'])!s:>6}.{str(info['time'] - int(info['time']))[2:3]} s | "
            f"{round(self.total_reward, 2)!s:<6}"
        )

        elapsed_time = time.time() - self.start_time

        if terminated:
            print("Game over")
            print(f"Elapsed time (real): {elapsed_time:.1f} seconds")
            print(f"Elapsed time (game): {info['time']:.1f} seconds")
            print(f"Distance ran: {info['distance']:.1f} m")
            print(f"Average speed: {info['avgspeed']:.1f} m/s")
            print(f"FPS: {self.n_steps / elapsed_time:.1f}")

        self.last_distance = info["distance"]
        self.last_time = info["time"]

        return obs, reward, terminated, truncated, info

    # Convenience methods for controlling verbosity
    def disable_verbose_wrapper(self):
        self.enabled = False

    def enable_verbose_wrapper(self):
        self.enabled = True

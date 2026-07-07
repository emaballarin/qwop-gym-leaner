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
import os

import gymnasium as gym

__all__: list[str] = ["RecordWrapper"]


class RecordWrapper(gym.Wrapper):
    def __init__(self, env, rec_file, overwrite, max_time, min_distance):
        super().__init__(env)

        if os.path.exists(rec_file) and not overwrite:
            raise Exception(f"rec_file already exists: {rec_file}")

        os.makedirs(os.path.dirname(rec_file), exist_ok=True)

        print(f"Recording to {rec_file}")
        self.handle = open(rec_file, "w")
        self.handle.write(f"seed={env.unwrapped.seedval}\n")
        self.max_time = max_time or 999
        self.min_distance = min_distance or 0
        self.actions = []
        self.discarded_episodes = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(str(action))

        if terminated:
            ep_info = f"episode with time={info['time']:.2f}"
            ep_info += f" and distance={info['distance']:.2f}"

            if info["time"] > self.max_time:
                print(f"Discarded {ep_info} (max_time exceeded)")
                self.discarded_episodes.append(self.actions)
            elif info["distance"] < self.min_distance:
                print(f"Discarded {ep_info} (min_distance not reached)")
                self.discarded_episodes.append(self.actions)
            else:
                # Dump discarded episodes only when there is
                # a regular episode after them
                if len(self.discarded_episodes) > 0:
                    print(f"Dump {len(self.discarded_episodes)} discarded episodes")
                    episodes = [ep + ["X"] for ep in self.discarded_episodes]
                    actions = [a for ep in episodes for a in ep]
                    self.handle.write("\n".join(actions) + "\n")
                    self.discarded_episodes = []

                self.handle.write("\n".join(self.actions) + "\n*\n")
                print(f"Recorded {ep_info}")

            self.actions = []

        return obs, reward, terminated, truncated, info

    def close(self):
        """Close the recording file and clean up resources."""
        if hasattr(self, "handle") and self.handle and not self.handle.closed:
            self.handle.flush()
            self.handle.close()
        super().close()

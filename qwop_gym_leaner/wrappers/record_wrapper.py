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
    """Record the actions of completed episodes, for `scripts/replay_recording.py`.

    File format: `key=value` header lines (seed, frames_per_step, reduced_action_set),
    then one action per line, each episode followed by a `*` line. Only terminated
    episodes within `max_time` and `min_distance` are written; truncated or interrupted
    ones are dropped. Episodes do not depend on each other, so each replays on its own.
    """

    def __init__(self, env, rec_file, overwrite=False, max_time=None, min_distance=None):
        super().__init__(env)

        if os.path.exists(rec_file) and not overwrite:
            raise FileExistsError(f"rec_file already exists: {rec_file}")

        dirname = os.path.dirname(rec_file)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

        print(f"Recording to {rec_file}")
        base = env.unwrapped
        self.handle = open(rec_file, "w", encoding="utf-8")  # noqa: SIM115 - written across steps, closed in close()
        self.handle.write(
            f"seed={base.seedval}\n"
            f"frames_per_step={base.frames_per_step}\n"
            f"reduced_action_set={int(base.reduced_action_set)}\n"
        )
        self.max_time = 999 if max_time is None else max_time
        self.min_distance = 0 if min_distance is None else min_distance
        self.actions = []

    def reset(self, *, seed=None, options=None):
        if self.actions:
            print(f"Discarded episode interrupted after {len(self.actions)} steps")
        self.actions = []
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.actions.append(str(int(action)))

        if terminated or truncated:
            ep_info = f"episode with time={info['time']:.2f}"
            ep_info += f" and distance={info['distance']:.2f}"

            if not terminated:
                print(f"Discarded {ep_info} (truncated)")
            elif info["time"] > self.max_time:
                print(f"Discarded {ep_info} (max_time exceeded)")
            elif info["distance"] < self.min_distance:
                print(f"Discarded {ep_info} (min_distance not reached)")
            else:
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

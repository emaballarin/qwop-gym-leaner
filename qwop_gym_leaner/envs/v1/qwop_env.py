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
import functools
import itertools
import math
import numbers
import os
import shutil
import socket
import struct
import weakref
from typing import ClassVar

import gymnasium as gym
import numpy as np

from .util.log import Log
from .util.wsclient import WSClient
from .util.wsclient import WSClientMock
from .util.wsproto import ProtocolError
from .util.wsproto import to_bytes
from .util.wsproto import WSProto
from .util.wsserver import RELOAD_TIMEOUT
from .util.wsserver import WSServer

__all__: list[str] = ["QwopEnv"]

BYTES_RESET = to_bytes(WSProto.H_CMD) + to_bytes(WSProto.CMD_RST)
BYTES_RELOAD = to_bytes(WSProto.H_RLD)
INT_OBS = int(WSProto.H_OBS)
OBS_PACKET_LEN = 2 + 4 * (2 + 60)  # header + flags + float32 time, distance and 60 observation values
SEED_MAX = np.iinfo(np.int32).max

# the numpy data type
# it seems pytorch is optimized for float32
DTYPE = np.float32


class Reaction:
    def __init__(self, flags, time, distance, data, ndata):
        self.data = data
        self.ndata = ndata
        self.time = time
        self.distance = distance
        self.game_over = bool(flags & WSProto.OBS_END)

        # NOTE: Normally, `OBS_SUC` and `OBS_END` are both set whenever the
        #       athlete steps beyond the finish line (100 meters) or if the
        #       athlete reaches 105 meters without stepping at all.
        self.is_success = bool(flags & WSProto.OBS_SUC)


class Normalizable:
    def __init__(self, name, limit_min, limit_max):
        self.name = name
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.center = DTYPE((limit_min + limit_max) / 2)
        self.maxdev = DTYPE(limit_max - self.center)

    def normalize(self, value):
        return (value - self.center) / self.maxdev


def _check_seed(seed):
    if not 0 <= seed <= SEED_MAX:
        raise ValueError(f"seed must be between 0 and {SEED_MAX}, got: {seed}")


def _resolve_executable(path, default, argname):
    """Return `path` (a file, or a command looked up on PATH; `default` if None) checked to be executable."""
    if path is None:
        resolved = shutil.which(default)
        if resolved is None:
            raise ValueError(f"no `{argname}` given and '{default}' is not available in PATH")
        return resolved
    resolved = shutil.which(path)  # an executable file as given, else a command found on PATH
    if resolved is None or not os.path.isfile(resolved):
        raise ValueError(f"`{argname}` is neither an executable file nor a command on PATH: {path}")
    return resolved


def _close(client, server):
    """Close the connection, then stop the relay thread (which quits the browser)."""
    try:
        client.close()
    finally:
        if server is not None:
            server.stop()


class QwopEnv(gym.Env):
    """
    A Gymnasium environment for Bennett Foddy's game QWOP (numeric state only).

    The game runs in a Chrome browser driven by selenium, relayed through a websocket server
    that runs in a daemon thread of the env's process. Each episode is a deterministic function
    of its own actions: neither previous episodes nor the seed affect it (the seed only re-seeds
    the page's `Math.random`, which the physics never uses), for a given Chrome build.

    browser: Path or command name of the web browser executable (defaults to 'chrome_for_testing' from PATH).
    driver: Path or command name of the chromedriver executable (defaults to 'chromedriver_for_testing' from PATH).
    failure_cost: Subtracted from the reward at the end of unsuccessful episodes.
    success_reward: Added to the reward at the end of successful episodes.
    time_cost_mult: Multiplier for the amount subtracted from the reward at each step.
    frames_per_step: Number of frames to advance per call to `.step` (aka. frameskip).
    stat_in_browser: Display a table with various game stats in the browser.
    game_in_browser: Display (and render, at every step) the game in a browser window;
        if False, the browser runs headless and nothing is rendered.
    text_in_browser: Display a static text next to the game area in the browser.
    reload_on_reset: Perform a page reload on each call to `.reset` (aka. "hard reset").
    reduced_action_set: Reduce possible actions from 16 to just 9:
        Genuine set: (none),Q,W,O,P,QW,QO,QP,WO,WP,OP,QWO,QWP,QOP,WOP,QWOP
        Reduced set: (none),Q,W,O,P,QW,QP,WO,OP.
    max_episode_time: End the episode as truncated once the game clock reaches this many
        game-seconds (300 frames each), whatever `frames_per_step` is; None disables the limit.
    loglevel: Logger level (DEBUG|INFO|WARN|ERROR).
    seed: Initial seed for the page's RNG (random if None); passing any seed to `.reset` reloads the page.
    browser_mock: Answer every command with a fixed observation instead of running the game (for tests).
    """

    metadata: ClassVar[dict] = {}

    def __init__(
        self,
        browser: str | None = None,
        driver: str | None = None,
        failure_cost=10,
        success_reward=50,
        time_cost_mult=10,
        frames_per_step=1,
        stat_in_browser=False,
        game_in_browser=True,
        text_in_browser=None,
        reload_on_reset=False,
        reduced_action_set=False,
        max_episode_time=180,
        loglevel="WARN",
        seed=None,
        browser_mock=False,
    ):
        seedval = seed if seed is not None else np.random.default_rng().integers(2**31)
        _check_seed(seedval)
        self.seedval = int(seedval)

        if isinstance(frames_per_step, bool) or not (
            isinstance(frames_per_step, numbers.Integral) and frames_per_step >= 1
        ):
            raise ValueError(f"frames_per_step must be a positive integer, got: {frames_per_step!r}")
        self.frames_per_step = frames_per_step = int(frames_per_step)

        if max_episode_time is not None:
            if isinstance(max_episode_time, bool) or not isinstance(max_episode_time, numbers.Real):
                raise ValueError(
                    f"max_episode_time must be a number of game-seconds or None, got: {max_episode_time!r}"
                )
            try:
                max_episode_time = float(max_episode_time)
            except OverflowError:
                max_episode_time = math.inf
            if not 0 < max_episode_time < math.inf:
                raise ValueError(f"max_episode_time must be positive and finite, got: {max_episode_time!r}")
        self.max_episode_time = max_episode_time
        self.game_in_browser = game_in_browser

        if browser_mock:
            self.server = None
            self.client = WSClientMock()
        else:
            browser = _resolve_executable(browser, "chrome_for_testing", "browser")
            driver = _resolve_executable(driver, "chromedriver_for_testing", "driver")

            # Listening before the relay starts means the client's connection is queued, never refused
            sock = socket.socket()
            try:
                sock.bind(("127.0.0.1", 0))
                sock.listen()
                self.server = WSServer(
                    sock,
                    seed=self.seedval,
                    stepsize=frames_per_step,
                    stat_in_browser=stat_in_browser,
                    game_in_browser=game_in_browser,
                    text_in_browser=text_in_browser,
                    driver=driver,
                    browser=browser,
                    loglevel=loglevel,
                )
            except BaseException:
                sock.close()
                raise

            self.server.start()
            try:
                self.client = WSClient(self.server, loglevel)
            except BaseException:
                self.server.stop()
                raise

        # Runs on .close() or, at the latest, at interpreter exit, so the browser never outlives the env
        self._finalizer = weakref.finalize(self, _close, self.client, self.server)

        self.reload_on_reset = reload_on_reset
        self.logger = Log.get_logger(__name__, loglevel)

        self.reduced_action_set = reduced_action_set
        self._set_keycodes()

        self.action_space = gym.spaces.Discrete(len(self.action_cmdflags))
        self.observation_space = gym.spaces.Box(shape=(60,), low=-1, high=1, dtype=DTYPE)

        self.speed_rew_mult = DTYPE(0.01)
        self.time_cost_mult = DTYPE(time_cost_mult)
        self.failure_cost = DTYPE(failure_cost)
        self.success_reward = DTYPE(success_reward)

        self.pos_x = Normalizable("pos_x", DTYPE(-10), DTYPE(1050))
        self.pos_y = Normalizable("pos_y", DTYPE(-10), DTYPE(10))
        self.angle = Normalizable("angle", DTYPE(-6), DTYPE(6))
        self.vel_x = Normalizable("vel_x", DTYPE(-20), DTYPE(60))
        self.vel_y = Normalizable("vel_y", DTYPE(-25), DTYPE(60))

        zeros = np.zeros(self.observation_space.shape, dtype=DTYPE)
        self.noop_reaction = Reaction(0, 0, 0, zeros, zeros)
        self._reset_env(self.noop_reaction)

        self.logger.info("Initialized with seed: %d", self.seedval)

    def _set_keycodes(self):
        keymap = {
            "q": WSProto.CMD_K_Q,
            "w": WSProto.CMD_K_W,
            "o": WSProto.CMD_K_O,
            "p": WSProto.CMD_K_P,
        }

        # All possible key combinations: (none), Q, W, O, P, QW, QO, ..., QWOP
        keys_c = [c for n in range(len(keymap) + 1) for c in itertools.combinations(keymap, n)]

        if self.reduced_action_set:
            # Remove useless key combinations instead of
            # letting agents work it out on their own
            redundant_combinations = {
                ("q", "o"),
                ("w", "p"),
                ("q", "w", "o"),
                ("q", "w", "p"),
                ("q", "o", "p"),
                ("w", "o", "p"),
                ("q", "w", "o", "p"),
            }
            keys_c = [c for c in keys_c if c not in redundant_combinations]

        # Key combinations represented as WSProto cmdflags
        self.action_cmdflags = [functools.reduce(lambda a, k: a | keymap[k], c, 0) for c in keys_c]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        needs_reload = self.reload_on_reset

        if seed is not None:
            # QWOP's seed can be changed ONLY if reloading the page
            _check_seed(seed)
            self.seedval = seed
            needs_reload = True

        try:
            reaction = self._restart_game(reload_page=needs_reload)
            self._reset_env(reaction)
        except BaseException as e:
            self.client.poison(e)  # the page may have reset without the env knowing
            raise
        return reaction.ndata, self._build_info(reaction)

    def _reset_env(self, reaction):
        self.steps = 0
        self.last_reaction = reaction
        self.last_reward = DTYPE(0)
        self.total_reward = DTYPE(0)

    def _restart_game(self, reload_page=False):
        if reload_page:
            data = self.client.send(BYTES_RELOAD + to_bytes(self.seedval, 4), timeout=RELOAD_TIMEOUT + 5)
            if data[0] != WSProto.H_ACK:
                raise ProtocolError(f"expected an ACK header, got: {data[0]}")

        return self._build_reaction(self.client.send(BYTES_RESET))

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r} for {self.action_space}")
        self.steps += 1

        try:
            reaction = self._perform_action(action)
            reward = self._calc_reward(reaction, self.last_reaction)
            terminated = reaction.game_over
            truncated = bool(
                not terminated and self.max_episode_time is not None and reaction.time >= self.max_episode_time
            )
            info = self._build_info(reaction)

            self.last_reward = reward
            self.total_reward += reward
            self.last_reaction = reaction
        except BaseException as e:
            # e.g. Ctrl-C after the page applied the step: the env's state no longer matches the page's
            self.client.poison(e)
            raise

        return reaction.ndata, reward, terminated, truncated, info

    def _perform_action(self, action):
        cmdflags = WSProto.CMD_STP | self.action_cmdflags[action]

        # Add draw flag if game rendering is enabled
        if self.game_in_browser:
            cmdflags |= WSProto.CMD_DRW

        data = (
            to_bytes(WSProto.H_CMD)
            + to_bytes(cmdflags)
            + to_bytes(self.steps & 0xFFFF, 2)  # uint16 on the wire, only displayed by the page
            + struct.pack("=f", self.last_reward)
            + struct.pack("=f", self.total_reward)
        )

        resp = self.client.send(data)
        return self._build_reaction(resp)

    def _build_reaction(self, data):
        if data[0] != INT_OBS or len(data) != OBS_PACKET_LEN:
            raise ProtocolError(
                f"expected a {OBS_PACKET_LEN}-byte OBS packet, got header {data[0]} and {len(data)} bytes"
            )

        flags = data[1]
        floats = np.frombuffer(data[2:], dtype=DTYPE)
        time = floats[0]
        distance = floats[1]
        obsdata = floats[2:]  # 60 floats (12 bodyparts, 5 floats per part)
        nobsdata = self._normalize(obsdata)
        return Reaction(flags, time, distance, obsdata, nobsdata)

    def _normalize(self, obs):
        length = len(obs)
        nobs = np.zeros(length, dtype=DTYPE)

        for i in range(0, len(obs), 5):
            nobs[i] = self.pos_x.normalize(obs[i])
            nobs[i + 1] = self.pos_y.normalize(obs[i + 1])
            nobs[i + 2] = self.angle.normalize(obs[i + 2])
            nobs[i + 3] = self.vel_x.normalize(obs[i + 3])
            nobs[i + 4] = self.vel_y.normalize(obs[i + 4])

        # Clamp values to -1..1
        np.clip(nobs, -1, 1, out=nobs)
        return nobs

    # r = reaction, lr = last_reaction
    def _calc_reward(self, reaction, last_reaction):
        ds = reaction.distance - last_reaction.distance
        dt = reaction.time - last_reaction.time
        # dt can be 0 for degenerate (e.g. repeated) observations; treat as no motion
        v = ds / dt if dt != 0 else DTYPE(0)
        rew = v * self.speed_rew_mult - dt * self.time_cost_mult / self.frames_per_step

        if reaction.game_over:
            if reaction.is_success:
                rew += self.success_reward
            else:
                rew -= self.failure_cost

        return rew

    def _build_info(self, reaction):
        # time is ~0 right after a reset; avoid a 0/0 nan leaking into info
        avgspeed = reaction.distance / reaction.time if reaction.time != 0 else DTYPE(0)
        return {
            "time": reaction.time,
            "distance": reaction.distance,
            "avgspeed": avgspeed,
            "is_success": reaction.is_success,
        }

    def close(self):
        self._finalizer()

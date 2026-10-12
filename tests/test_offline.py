"""Tests that need no browser: the env against its mock client, the wrappers and the recording format."""

import functools
import importlib.util
import pathlib
import struct
import subprocess
import sys
import textwrap
import warnings

import gymnasium as gym
import numpy as np
import pytest

from qwop_gym_leaner import AbsorbWrapper
from qwop_gym_leaner import QwopEnv
from qwop_gym_leaner import RecordWrapper
from qwop_gym_leaner import VerboseWrapper
from qwop_gym_leaner.envs.v1.util.wsproto import to_bytes
from qwop_gym_leaner.envs.v1.util.wsproto import WSProto

REPO = pathlib.Path(__file__).resolve().parents[1]
KEYS = {"Q": WSProto.CMD_K_Q, "W": WSProto.CMD_K_W, "O": WSProto.CMD_K_O, "P": WSProto.CMD_K_P}


def obs_packet(time, distance, flags=0):
    return (
        to_bytes(WSProto.H_OBS)
        + to_bytes(flags)
        + struct.pack("=ff", time, distance)
        + np.zeros(60, dtype=np.float32).tobytes()
    )


class ScriptedClient:
    """Replies to each command with the next packet of a script, and records what was sent and poisoned."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.poisoned = []

    def send(self, data, timeout=None):
        self.sent.append(data)
        return self.replies.pop(0)

    def poison(self, error):
        self.poisoned.append(error)

    def close(self):
        pass


def load_replay_script():
    spec = importlib.util.spec_from_file_location("replay_recording", REPO / "scripts" / "replay_recording.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("reduced", "names"),
    [
        (False, ["", "Q", "W", "O", "P", "QW", "QO", "QP", "WO", "WP", "OP", "QWO", "QWP", "QOP", "WOP", "QWOP"]),
        (True, ["", "Q", "W", "O", "P", "QW", "QP", "WO", "OP"]),
    ],
)
def test_action_indices_map_to_documented_key_combinations(reduced, names):
    # recordings and trained policies depend on these indices
    env = QwopEnv(browser_mock=True, reduced_action_set=reduced)
    assert env.action_space.n == len(names)
    assert env.action_cmdflags == [sum(KEYS[k] for k in name) for name in names]


@pytest.mark.parametrize(("reduced", "action"), [(False, -1), (False, 16), (True, 9)])
def test_step_rejects_actions_outside_the_action_space(reduced, action):
    env = QwopEnv(browser_mock=True, reduced_action_set=reduced)
    env.reset()
    with pytest.raises(ValueError, match="invalid action"):
        env.step(action)


def test_step_counter_wraps_on_the_wire_instead_of_overflowing():
    env = QwopEnv(browser_mock=True)
    env.reset()
    env.steps = 65535
    env.client = ScriptedClient([obs_packet(1.0, 1.0)])
    env.step(0)
    assert env.steps == 65536
    assert int.from_bytes(env.client.sent[0][2:4], sys.byteorder) == 0


def test_first_step_reward_is_measured_from_the_reset_state():
    env = QwopEnv(browser_mock=True, frames_per_step=2)
    env.client = ScriptedClient([obs_packet(0.5, 3.0), obs_packet(0.5 + 2 / 300, 3.01)])
    env.reset()
    _, reward, *_ = env.step(0)
    dt = np.float32(0.5 + 2 / 300) - np.float32(0.5)
    ds = np.float32(3.01) - np.float32(3.0)
    assert reward == pytest.approx(ds / dt * 0.01 - dt * 10 / 2, rel=1e-6)


def interrupt(*args):
    raise KeyboardInterrupt


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("malformed step reply", "ProtocolError"),
        ("malformed reset reply", "ProtocolError"),
        ("interrupt after the reply", "KeyboardInterrupt"),
    ],
)
def test_a_failure_after_a_command_was_sent_poisons_the_client(case, error):
    # the page may have applied the command: no later command may run against it
    from qwop_gym_leaner.envs.v1.util.wsproto import ProtocolError

    env = QwopEnv(browser_mock=True)
    ack = bytes([WSProto.H_ACK])
    if case == "malformed reset reply":
        env.client = ScriptedClient([ack])
        call = env.reset
    else:
        step_reply = ack if case == "malformed step reply" else obs_packet(0.01, 0.26)
        env.client = ScriptedClient([obs_packet(0.0067, 0.25), step_reply])
        env.reset()
        if case == "interrupt after the reply":
            env._normalize = interrupt
        call = functools.partial(env.step, 0)
    with pytest.raises({"ProtocolError": ProtocolError, "KeyboardInterrupt": KeyboardInterrupt}[error]):
        call()
    assert [type(e).__name__ for e in env.client.poisoned] == [error]


@pytest.mark.parametrize(
    ("kwargs", "time", "flags", "expected"),
    [
        ({}, 179.99, 0, (False, False)),  # default limit: 180 game-seconds
        ({}, 180.0, 0, (False, True)),
        ({}, 180.0, WSProto.OBS_END, (True, False)),  # the game ended first
        ({"max_episode_time": 10}, 10.01, 0, (False, True)),
        ({"max_episode_time": None}, 1e6, 0, (False, False)),
    ],
)
def test_episodes_are_truncated_at_the_game_time_limit(kwargs, time, flags, expected):
    env = QwopEnv(browser_mock=True, **kwargs)
    env.client = ScriptedClient([obs_packet(0.0067, 0.25), obs_packet(time, 1.0, flags)])
    env.reset()
    _, _, terminated, truncated, _ = env.step(0)
    assert (terminated, truncated) == expected
    assert type(truncated) is bool  # gymnasium's contract, not numpy's bool


@pytest.mark.parametrize("value", [0, -5, "180", float("nan"), 10**400])
def test_invalid_game_time_limits_are_rejected(value):
    with pytest.raises(ValueError, match="max_episode_time"):
        QwopEnv(browser_mock=True, max_episode_time=value)


def test_mock_env_passes_the_gymnasium_checker():
    from gymnasium.utils.env_checker import check_env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(QwopEnv(browser_mock=True), skip_render_check=True)


def test_validation_survives_python_O():
    # experiments run under `python -O`, which strips every assert
    script = textwrap.dedent(
        """
        import struct
        from qwop_gym_leaner import QwopEnv
        from qwop_gym_leaner.envs.v1.util.wsproto import WSProto

        def outcome(fn):
            try:
                fn()
            except Exception as e:
                return type(e).__name__
            return "accepted"

        class Client:
            def __init__(self, reply):
                self.reply = reply
            def send(self, data, timeout=None):
                return self.reply
            def poison(self, error):
                pass
            def close(self):
                pass

        def with_client(reply, call):
            env = QwopEnv(browser_mock=True)
            env.client = Client(reply)
            call(env)

        ack = bytes([WSProto.H_ACK])
        obs = bytes([WSProto.H_OBS, 0]) + struct.pack("=ff", 1, 1) + bytes(240)
        checks = {
            "negative seed": (lambda: QwopEnv(browser_mock=True, seed=-5), "ValueError"),
            "seed above int32": (lambda: QwopEnv(browser_mock=True, seed=2**31), "ValueError"),
            "zero frames_per_step": (lambda: QwopEnv(browser_mock=True, frames_per_step=0), "ValueError"),
            "ACK where OBS expected": (lambda: with_client(ack, lambda e: e.step(0)), "ProtocolError"),
            "truncated OBS": (lambda: with_client(obs[:-4], lambda e: e.step(0)), "ProtocolError"),
            "OBS where reload ACK expected": (lambda: with_client(obs, lambda e: e.reset(seed=1)), "ProtocolError"),
        }
        failures = [f"{k}: {outcome(f)} (expected {want})" for k, (f, want) in checks.items() if outcome(f) != want]
        print("\\n".join(failures))
        raise SystemExit(1 if failures else 0)
        """
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", script], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("case", ["missing browser", "non-executable driver", "not on PATH"])
def test_bad_executables_are_rejected_before_anything_starts(case, tmp_path, monkeypatch):
    plain = tmp_path / "chromedriver"
    plain.write_text("")
    kwargs = {
        "missing browser": {"browser": str(tmp_path / "nope"), "driver": sys.executable},
        "non-executable driver": {"browser": sys.executable, "driver": str(plain)},
        "not on PATH": {},
    }[case]
    if case == "not on PATH":
        monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(ValueError, match=r"browser|driver"):
        QwopEnv(**kwargs)


# ---------------------------------------------------------------------- wrappers


class EpisodeEnv(gym.Env):
    """Minimal QWOP-like env: each episode terminates after `ep_len` steps, distance 0.1 per step."""

    metadata = {}  # noqa: RUF012 - gymnasium convention
    action_space = gym.spaces.Discrete(16)
    observation_space = gym.spaces.Box(-1, 1, (60,), np.float32)
    seedval, frames_per_step, reduced_action_set = 42, 2, False

    def __init__(self, ep_len, truncate_at=None):
        self.ep_len = ep_len
        self.truncate_at = truncate_at  # cut the episode off itself, as the env's time limit does
        self.action_cmdflags = QwopEnv(browser_mock=True).action_cmdflags

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.zeros(60, np.float32), {"time": np.float32(0), "distance": np.float32(0)}

    def step(self, action):
        self.t += 1
        info = {"time": np.float32(self.t / 30), "distance": np.float32(self.t / 10), "avgspeed": 3.0}
        terminated = self.t >= self.ep_len
        truncated = not terminated and self.truncate_at is not None and self.t >= self.truncate_at
        return np.full(60, self.t / 100, np.float32), np.float32(-10 if terminated else 1), terminated, truncated, info


def test_record_wrapper_writes_only_completed_episodes(tmp_path):
    rec = tmp_path / "sub" / "rec.txt"
    env = RecordWrapper(EpisodeEnv(ep_len=3), str(rec), overwrite=False, max_time=None, min_distance=None)
    limited = gym.wrappers.TimeLimit(env, max_episode_steps=2)
    limited.reset()
    limited.step(1), limited.step(2)  # cut off by an outer TimeLimit
    limited.reset()
    limited.step(3)
    env.reset()  # interrupted after one step
    for action in (4, 5, 6):
        env.step(action)  # terminated
    env.close()
    assert rec.read_text().splitlines() == [
        "seed=42",
        "frames_per_step=2",
        "reduced_action_set=0",
        "4",
        "5",
        "6",
        "*",
    ]


def test_record_wrapper_drops_episodes_the_env_truncates(tmp_path):
    # e.g. QwopEnv's own max_episode_time, inside the wrapper
    rec = tmp_path / "rec.txt"
    inner = EpisodeEnv(ep_len=3, truncate_at=2)
    env = RecordWrapper(inner, str(rec), overwrite=False, max_time=None, min_distance=None)
    env.reset()
    env.step(1), env.step(2)  # truncated by the env
    inner.truncate_at = None
    env.reset()
    for action in (3, 4, 5):
        env.step(action)
    env.close()
    assert rec.read_text().splitlines()[3:] == ["3", "4", "5", "*"]


def test_record_wrapper_accepts_a_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    RecordWrapper(EpisodeEnv(ep_len=3), "rec.txt", overwrite=False, max_time=None, min_distance=None).close()
    assert (tmp_path / "rec.txt").exists()


@pytest.mark.parametrize(
    ("text", "header", "episodes"),
    [
        (
            "seed=7\nframes_per_step=2\nreduced_action_set=1\n1\n2\n*\n3\n*\n",
            {"seed": 7, "frames_per_step": 2, "reduced_action_set": 1},
            [[1, 2], [3]],
        ),
        ("seed=7\n1\n1\nX\n2\n2\n*\n", {"seed": 7}, [[2, 2]]),  # legacy: discarded `X` blocks are skipped
    ],
)
def test_parse_recording_returns_header_and_every_episode(tmp_path, text, header, episodes):
    rec = tmp_path / "rec.txt"
    rec.write_text(text)
    assert load_replay_script().parse_recording(str(rec)) == (header, episodes)


@pytest.mark.parametrize(
    "text", ["1\n*\n", "seed=7\n1\nfoo\n*\n", "seed=7\n16\n*\n", "seed=7\n1\n2\n", "seed=7\n1\n*\nseed=8\n"]
)
def test_parse_recording_rejects_malformed_files(tmp_path, text):
    rec = tmp_path / "rec.txt"
    rec.write_text(text)
    with pytest.raises(ValueError):
        load_replay_script().parse_recording(str(rec))


def test_verbose_wrapper_works_on_a_registered_env(capsys):
    env = VerboseWrapper(gym.make("QWOP-v2", browser_mock=True))
    env.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)  # dt == 0 between the mock's identical observations
        env.step(0)
        env.step(0)
    assert "...." in capsys.readouterr().out


def test_the_superseded_env_id_is_refused():
    # results from QWOP-v1 are not comparable with this version's
    with pytest.raises(gym.error.DeprecatedEnv, match="QWOP-v2"):
        gym.make("QWOP-v1", browser_mock=True)


def test_absorb_wrapper_repeats_the_terminal_step_without_terminating():
    env = AbsorbWrapper(EpisodeEnv(ep_len=2))
    env.reset()
    env.step(0)
    terminal = env.step(0)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(0)
        assert terminated is False and truncated is False
        np.testing.assert_array_equal(obs, terminal[0])
        assert reward == terminal[1] == -10
        assert info == terminal[4]

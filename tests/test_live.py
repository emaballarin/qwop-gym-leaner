"""Tests against the real game in headless Chrome for Testing (`pytest -m live`)."""

import os
import subprocess
import sys
import textwrap
import time
import warnings

import gymnasium as gym
import numpy as np
import pytest

pytestmark = pytest.mark.live


def actions(n, key, nact=16):
    return np.random.default_rng(np.random.SeedSequence(key)).integers(nact, size=n).tolist()


def gait(a, b, period, n):
    """Alternate actions `a` and `b` every `period` steps: a crude but long-distance run."""
    return [a if (t // period) % 2 == 0 else b for t in range(n)]


def run_episode(env, acts, **reset_kwargs):
    """Reset, then play `acts` until termination; return raw observations, rewards and game times."""
    _, info = env.reset(**reset_kwargs)
    obs, rewards, times = [env.unwrapped.last_reaction.data.copy()], [], [info["time"]]
    for a in acts:
        _, reward, terminated, _, info = env.step(a)
        obs.append(env.unwrapped.last_reaction.data.copy())
        rewards.append(reward)
        times.append(info["time"])
        if terminated:
            break
    return np.stack(obs), np.array(rewards), np.array(times)


def browser_processes(binaries):
    """PIDs of running processes whose executable is one of `binaries` (Linux only)."""
    targets = {os.path.realpath(b) for b in binaries}
    pids = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if os.readlink(f"/proc/{pid}/exe") in targets:
                pids.append(int(pid))
        except OSError:
            continue
    return pids


@pytest.mark.parametrize(
    "probe",
    [
        # ~54 m; fails without the broadphase rebuild (successive replays alternate)
        gait(8, 7, 8, 4000),
        # fails without the track reset
        actions(1500, 20261011),
    ],
    ids=["long-distance gait", "random actions"],
)
def test_episodes_depend_only_on_their_own_actions(make_env, probe):
    env = make_env(seed=777)
    reference = run_episode(env, probe, seed=777)
    histories = {
        "the same episode": [probe],
        "terminated short episodes": [actions(300, 9 + k) for k in range(3)],
        "a no-op episode": [[0] * 2500],
    }

    def check(result, where):
        for got, want, what in zip(result, reference, ("observations", "rewards", "game times"), strict=True):
            np.testing.assert_array_equal(got, want, err_msg=f"{what} {where}")

    # every replay of the probe must match, including those run as history
    for name, history in histories.items():
        for h in history:
            result = run_episode(env, h)
            if h is probe:
                check(result, "on the first soft reset")
        check(run_episode(env, probe), f"after {name}")

    # the seed only re-seeds Math.random, which the physics never uses
    check(run_episode(make_env(seed=123), probe), "in another env instance with another seed")


def test_live_env_passes_the_gymnasium_checker(make_env):
    from gymnasium.utils.env_checker import check_env

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(make_env(), skip_render_check=True)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="process inspection via /proc")
def test_no_browser_process_outlives_the_env(browser_paths, make_env):
    before = set(browser_processes(browser_paths))
    env = make_env()
    env.reset()
    env.close()
    time.sleep(1)
    assert set(browser_processes(browser_paths)) <= before, "processes left after close()"

    # an env that is never closed must neither block interpreter exit nor leave orphans behind
    script = textwrap.dedent(
        f"""
        from qwop_gym_leaner import QwopEnv
        env = QwopEnv(browser={browser_paths[0]!r}, driver={browser_paths[1]!r}, game_in_browser=False)
        env.reset()
        env.step(0)
        """
    )
    subprocess.run([sys.executable, "-c", script], timeout=90, check=True)
    time.sleep(1)
    assert set(browser_processes(browser_paths)) <= before, "processes left after exiting without close()"


def test_envs_run_in_daemonic_vector_workers(browser_paths):
    # gymnasium's AsyncVectorEnv (like SB3's SubprocVecEnv) runs each env in a daemonic process;
    # the module-qualified id registers QWOP-v2 in workers that start fresh (forkserver, spawn)
    def make():
        return gym.make(
            "qwop_gym_leaner:QWOP-v2", browser=browser_paths[0], driver=browser_paths[1], game_in_browser=False
        )

    venv = gym.vector.AsyncVectorEnv([make, make])
    try:
        assert venv.unwrapped.daemon
        venv.reset(seed=1)
        for a in actions(50, 3):
            venv.step(np.array([a, a]))
    finally:
        venv.close()


def test_a_browser_that_fails_to_start_raises_promptly(browser_paths):
    from qwop_gym_leaner import QwopEnv

    start = time.monotonic()
    with pytest.raises(ConnectionError, match="the relay failed"):
        QwopEnv(browser="/bin/true", driver=browser_paths[1], game_in_browser=False)
    assert time.monotonic() - start < 5


def break_page(env):
    """Make the page throw while handling the next command (after it has stepped the game)."""
    env.unwrapped.server._driver.execute_script('ws.fn_observation = () => { throw new Error("injected failure"); };')


@pytest.mark.parametrize("trigger", ["timeout", "page error"])
def test_a_failed_command_poisons_the_connection(make_env, trigger):
    # neither a late reply nor a half-applied command may leak into the next command
    from qwop_gym_leaner.envs.v1.util.wsproto import to_bytes
    from qwop_gym_leaner.envs.v1.util.wsproto import WSProto

    env = make_env()
    env.reset()
    if trigger == "timeout":
        # the page answers 0.3 s late, after the env has given up: the relay must drop that reply and survive
        env.unwrapped.server._driver.execute_script(
            "const f = ws.fn_observation;"
            "ws.fn_observation = () => { const t = performance.now(); while (performance.now() - t < 300); return f(); };"
        )
        step = to_bytes(WSProto.H_CMD) + to_bytes(WSProto.CMD_STP) + bytes(10)
        with pytest.raises(ConnectionError, match="no reply"):
            env.unwrapped.client.send(step, timeout=0.05)
        time.sleep(0.6)
        assert env.unwrapped.server.error is None and env.unwrapped.server.is_alive()
    else:
        break_page(env)
        with pytest.raises(RuntimeError, match="injected failure"):
            env.step(0)
    with pytest.raises(ConnectionError, match="earlier failure"):
        env.step(0)


def test_an_env_dropped_after_a_failure_is_collected_and_closed(browser_paths):
    import gc
    import weakref

    from qwop_gym_leaner import QwopEnv

    env = QwopEnv(browser=browser_paths[0], driver=browser_paths[1], game_in_browser=False)
    env.reset()
    break_page(env)
    with pytest.raises(RuntimeError):
        env.step(0)
    server, ref = env.server, weakref.ref(env)
    del env
    gc.collect()
    assert ref() is None, "the failed env is still referenced"
    assert not server.is_alive(), "the finalizer did not stop the relay"


@pytest.mark.parametrize("frames_per_step", [1, 2])
def test_the_time_limit_counts_game_seconds_whatever_the_frameskip(make_env, frames_per_step):
    # 300 frames per game-second; reset already accounts for 2 frames
    env = make_env(max_episode_time=1, frames_per_step=frames_per_step)
    env.reset()
    steps, truncated = 0, False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        steps += 1
    assert steps == -(-298 // frames_per_step)

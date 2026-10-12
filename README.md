# QWOP Gym

A Gym environment for Bennett Foddy's game called _QWOP_.

[Give it a try](https://www.foddy.net/Athletics.html) and see why it's such a
good candidate for Reinforcement Learning :)

You should also check this [video](https://www.youtube.com/watch?v=2qNKjRwcx74) for a demo.

### Features

- A call to `.step()` advances exactly N game frames (configurable)
- Headless mode (`game_in_browser=False`): no window, no rendering, for training
- Fully deterministic: each episode depends only on its own actions
- State extraction for a slim observation of 60 `float32` values (240 bytes)
- Real-time visualization of various game stats (optional)
- Additional in-game controls for easier debugging

## Getting started

1. Install [Python](https://www.python.org/downloads/) 3.11 or higher
1. Download [Chrome for Testing and the matching chromedriver](https://googlechromelabs.github.io/chrome-for-testing/)
   (any Chrome-based browser works with a chromedriver of the same version)
1. Install the `qwop-gym-leaner` package:

```bash
pip install qwop-gym-leaner
```

Create an instance in your code:

```python
import gymnasium as gym
import qwop_gym_leaner  # importing registers the "QWOP-v2" env id

env = gym.make("QWOP-v2", browser="/browser/path", driver="/driver/path", game_in_browser=False)
```

`browser` and `driver` default to `chrome_for_testing` and
`chromedriver_for_testing` from `PATH`.

An episode ends when the game does (the athlete falls or reaches 100 m:
`terminated=True`) or when the game clock reaches `max_episode_time` game-seconds
(default 180, i.e. 54,000 frames, whatever `frames_per_step` is:
`truncated=True`); `max_episode_time=None` removes the limit.

> [!NOTE]
> With the default `game_in_browser=True` a browser window opens and the game is
> rendered at every step, which is several times slower. With
> `game_in_browser=False` the browser runs headless. Either way the game runs in
> that browser, so call `env.close()` when done; an env that is never closed is
> closed at interpreter exit.

Vectorised envs (gymnasium's `AsyncVectorEnv`, SB3's `SubprocVecEnv`) work. Use
the module-qualified id `"qwop_gym_leaner:QWOP-v2"` when the worker processes
start fresh (the `spawn` and `forkserver` start methods; `forkserver` is the
default on Linux from Python 3.14), so that each worker registers the env.

### Determinism

Every episode is a deterministic function of its own actions, for a given
Chrome build. Episodes do not depend on earlier episodes (soft resets restore
the complete physics state) nor on the seed: the seed only re-seeds the page's
`Math.random`, which the physics never uses. Replaying recorded actions
(`RecordWrapper`, then `scripts/replay_recording.py`) reproduces each episode
exactly.

## Tests

```bash
pip install -e ".[test]"
pytest          # offline tests, no browser needed
pytest -m live  # against the real game; set QWOP_BROWSER / QWOP_DRIVER if not on PATH
```

## Similar projects

- <https://github.com/Wesleyliao/QWOP-RL>
- <https://github.com/drakesvoboda/RL-QWOP>
- <https://github.com/juanto121/qwop-ai>
- <https://github.com/ShawnHymel/qwop-ai>

In comparison, qwop-gym-leaner offers several key features:

- the env is _performant_ - perfect for on-policy algorithms as observations
  can be collected at great speeds (more than 2000 observations/sec on an Apple
  M2 CPU - orders of magnitude faster than the other QWOP RL envs).
- the env is _deterministic_ - there are no race conditions and no hidden
  randomness. Replaying recorded actions produces the same result.
- the env has a _simple reward model_ and compared to other QWOP envs, it is
  less biased, eg. no special logic for stuff like _knee bending_,
  _low torso height_, _vertical movement_, etc.
- the env allows all 16 key combinations (including no key), other QWOP envs
  usually allow only the "useful" 8 key combinations.
- great results (fast, human-like running) achieved by RL agents trained
  entirely through self-play, without pre-recorded expert demonstrations
- QWOP's original JS source code is barely modified.

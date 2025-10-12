# QWOP Gym

A Gym environment for Bennet Foddy's game called _QWOP_.

[Give it a try](https://www.foddy.net/Athletics.html) and see why it's such a
good candidate for Reinforcement Learning :)

You should also check this [video](https://www.youtube.com/watch?v=2qNKjRwcx74) for a demo.

### Features

- A call to `.step()` advances exactly N game frames (configurable)
- Option to disable WebGL rendering for improved performance
- Is fully deterministic \*
- State extraction for a slim observation of 60 bytes
- Real-time visualization of various game stats (optional)
- Additional in-game controls for easier debugging

\* given the state includes the steps since last hard reset, see [♻️ Resetting](./doc/env.md#resetting)

## Getting started

1. Install [Python](https://www.python.org/downloads/) 3.10 or higher
1. Install a chrome-based web browser (Google Chrome, Brave, Chromium, etc.)
1. Download [chromedriver](https://googlechromelabs.github.io/chrome-for-testing/) 116.0 or higher
1. Install the `qwop-gym-leaner` package:

```bash
pip install qwop-gym-leaner
```

Create an instance in your code:

```python
import qwop_gym_leaner

env = gym.make("QWOP-v1", browser="/browser/path", driver="/driver/path")
```

## The `qwop-gym-leaner` tool

The `qwop-gym-leaner` executable is a handy command-line tool which makes it easy to
play, record and replay episodes, train agents and more.

Firstly, perform the initial setup:

```
qwop-gym-leaner bootstrap
```

Play the game (use Q, W, O, P keys):

```bash
qwop-gym-leaner play
```

Explore the other available commands:

```bash
$ qwop-gym-leaner -h
usage: qwop-gym-leaner [options] <action>

options:
  -h, --help  show this help message and exit
  -c FILE     config file, defaults to config/<action>.yml

action:
  benchmark         evaluate the actions/s achievable with this env
  bootstrap         perform initial setup
  help              print this help message
```

> [!WARNING]
> Although no rendering occurs during training, the browser window must remain
> open as the game is actually running at very high speeds behind the curtains.

## Similar projects

- <https://github.com/Wesleyliao/QWOP-RL>
- <https://github.com/drakesvoboda/RL-QWOP>
- <https://github.com/juanto121/qwop-ai>
- <https://github.com/ShawnHymel/qwop-ai>

In comparison, qwop-gym-leaner offers several key features:

- the env is _performant_ - perfect for on-policy algorithms as observations
  can be collected at great speeds (more than 2000 observations/sec on an Apple
  M2 CPU - orders of magnitute faster than the other QWOP RL envs).
- the env is _deterministic_ - there are no race conditions and randomness can
  be removed if desired. Replaying recorded actions produces the same result.
- the env has a _simple reward model_ and compared to other QWOP envs, it is
  less biased, eg. no special logic for stuff like _knee bending_,
  _low torso height_, _vertical movement_, etc.
- the env allows all possible key combinations (15), other QWOP envs usually
  allow only the "useful" 8 key combinations.
- great results (fast, human-like running) achieved by RL agents trained
  entirely through self-play, without pre-recorded expert demonstrations
- QWOP's original JS source code is barely modified.

## Caveats

The below list highlights some areas in which the project could use some
improvements:

- the OS may put some pretty rough restrictions on the web browser's rendering
  as soon as it's put in the background (on OS X at least). Ideally, the browser
  should run in a headless mode, but experimentation with headless, WebGL-capable
  web browsers (e.g. [Chrome](https://developer.chrome.com/blog/supercharge-web-ai-testing))
  has been unsucessful so far.

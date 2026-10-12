# Changelog

## 2.0.0 (unreleased)

### Fixed

- **Starts on Python 3.14.** The relay no longer calls `asyncio.get_event_loop()`
  without a loop, which raised in the server process.
- **Episodes are fully deterministic.** Soft resets now restore the leapfrogging
  track and rebuild Box2D's broadphase. Before, an episode could depend on earlier
  episodes: a 4,000-step run (about 54 m) replayed after another long run diverged
  at step 3,648. Reset no longer advances the game clock by wall-clock time.
- **The first step's reward is measured from the reset state.** It used to be
  measured from time 0 and distance 0, which gave a spurious, wall-clock-dependent
  value (observed from −0.24 to +0.29) in place of the usual ≈ −0.03. This also
  made the gymnasium `check_env` fail.
- **The env never hangs and never orphans the browser.**
    - A crashed relay or browser now raises promptly with its cause, chained;
      `QwopEnv()` used to retry forever.
    - Exiting without `close()` no longer blocks the interpreter or leaves Chrome
      running.
    - Commands are never silently re-sent after a connection loss.
- **Envs work in daemonic workers** (gymnasium `AsyncVectorEnv`, SB3
  `SubprocVecEnv`). The relay is now a thread, not a child process.
- **Startup no longer pays a fixed 5 s retry** for a connection race (measured 5.0 s → 2.3 s).
- **Validation no longer uses `assert`, so it survives `python -O`.**
    - Seed range, `frames_per_step`, protocol headers and packet length are checked
      explicitly.
    - Browser and driver executables are checked before anything starts.
    - Invalid actions raise `ValueError`; −1 used to silently map to the last
      action (QWOP, or OP with the reduced set).
- **No overflow after 65,535 steps.** The step counter (display only) wraps
  instead of raising `OverflowError`.
- **`RecordWrapper`**:
    - Actions of truncated or interrupted episodes no longer leak into the next
      recorded episode.
    - A bare filename works.
- **`VerboseWrapper`** works on `gym.make` envs, and no longer divides by zero.
- **`browser_mock=True`** supports `reset(seed=…)`, so `check_env` runs offline.
- The websockets ≥ 17.1 deprecation warning for `connect()` is gone, and one
  socket no longer leaks per env.

### Changed

- **The env id is now `QWOP-v2`** (package version 2.0.0). Rewards, dynamics and
  the default episode length changed, so results are not comparable with
  `QWOP-v1`; `gym.make("QWOP-v1")` raises gymnasium's `DeprecatedEnv`, pointing
  to `QWOP-v2`.
- **Episodes now end at 180 game-seconds by default** (`truncated=True`; see
  `max_episode_time` under Added). Pass `max_episode_time=None` for the old
  unlimited behaviour; `scripts/replay_recording.py` does.
- **The recording format** gains `frames_per_step` and `reduced_action_set`
  header lines. Only completed episodes that pass the filters are written; old
  versions also wrote `X` blocks of discarded episodes, needed as replay history.
- **`scripts/replay_recording.py`**:
    - Replays every episode, or the ones chosen with `--episode N`.
    - Takes the env parameters from the header.
    - Skips legacy `X` blocks and reports any episode that diverges from its
      recording.
    - Trajectories can differ from those of earlier versions, even for the first
      episode after a hard reset (in a 4,000-step run they first differ at step
      2,274), so episodes in legacy recordings may report a divergence.
- **A browser crash now raises `ConnectionError`.** The env used to relaunch
  Chrome and carry on, silently corrupting the episode in progress; recreate the
  env to recover.
- **Any failed or interrupted command closes the env's connection**: a timeout,
  a lost connection, Ctrl-C during `step()`, an error raised by the page, or a
  malformed reply. Later commands raise `ConnectionError` naming that first
  failure, so neither a late reply nor a half-applied command can leak into
  another command. Recreate the env to continue.
- `close()` (and interpreter exit) waits up to 30 s for the browser to quit.
- `browser_mock=True` reports a realistic game time (2 frames) and distance (the
  start) instead of a Unix timestamp, which the time limit would have truncated.
- Websocket compression is off (messages are 250 bytes).
- `RecordWrapper`'s `overwrite`, `max_time` and `min_distance` are optional, and
  an existing file raises `FileExistsError`.

### Removed

- Dead code:
    - the image-capture path (`CMD_IMG` / `H_IMG`) and `H_REJ`, `OBS_PAS`;
    - `manual_client`;
    - `Normalizable.denormalize`, `keycodes_c` and the unreachable verbose log
      branch;
    - `ws.js` `request()` and the unbound `escape` handler.

### Added

- **Episode time limit:** `QwopEnv(max_episode_time=180)` truncates an episode
  (`truncated=True`) once the game clock reaches that many game-seconds (300
  frames each, independent of `frames_per_step`); `None` disables it. Episodes
  used to run until the game ended, however long that took.
- A test suite: `pytest` runs the offline tests; `pytest -m live` runs against
  the real game.
- A CI workflow: offline on Python 3.11–3.14, live on 3.11 and 3.14.
- `PROJECT.md`.

# PROJECT.md

A Gymnasium environment for QWOP: the real game runs in Chrome (driven by
selenium) and the env talks to it over websockets. Fork of
[smanolloff/qwop-gym](https://github.com/smanolloff/qwop-gym) (the `upstream`
remote), stripped down to the env and its wrappers.

## Layout

- `qwop_gym_leaner/envs/v1/qwop_env.py`: `QwopEnv`: spaces, reward, normalisation, lifecycle.
- `qwop_gym_leaner/envs/v1/util/`
    - `wsserver.py`: `WSServer`, the relay. It runs in a daemon thread of the env's
      process with its own asyncio loop, and launches and quits the browser.
    - `wsclient.py`: `WSClient`, the env's synchronous end, plus `WSClientMock` for
      browser-free tests.
    - `wsproto.py`: wire-protocol constants and `ProtocolError`.
    - `log.py`: logger factory and message formatters.
- `qwop_gym_leaner/envs/v1/game/`: the page.
    - `QWOP.min.js` is upstream's instrumented build. It exposes `QWOP()` and the
      `QWOP.__i` class registry. Never hand-edit it.
    - `extensions.js` and `ws.js` are this project's glue: stepping, observation,
      reset, and the websocket peer.
- `qwop_gym_leaner/wrappers/`: `RecordWrapper`, `VerboseWrapper`, `AbsorbWrapper`.
- `scripts/replay_recording.py`: parses and replays `RecordWrapper` files.
- `tests/`: `test_offline.py` (default `pytest`) and `test_live.py` (`pytest -m live`, real Chrome).

## Architecture

```
main thread: QwopEnv -> WSClient  --ws-->  relay thread: WSServer  --ws-->  Chrome page: ws.js / extensions.js
```

- The env binds and `listen()`s on a socket first, so the client connects
  immediately.
- The relay acknowledges the env only once the page has registered.
- The first fatal relay error is stored in `WSServer.error` _before_ connections
  close. `WSClient` then re-raises it chained, and never re-sends a command.
- Any failed or interrupted command (transport failure, interruption, page error,
  malformed reply) closes the client's connection for good: later commands raise,
  naming that failure, and the env must be recreated. The relay drops replies
  that arrive for an env that has already given up.
- A `weakref.finalize` closes the client and stops the relay (which quits Chrome)
  on `close()`, or at the latest at interpreter exit.
- There is no multiprocessing, so envs can live in daemonic workers.

## Wire protocol

Constants in `wsproto.py` mirror `ws.js`: keep them in sync, and never renumber.

- Py → page: `H_CMD`, carrying cmdflags, a step counter and rewards. The counter
  is uint16, masked, and only displayed.
- Page → py: `H_OBS`, a fixed 250 bytes: header, flags, then `float32` time,
  distance and 60 observation values.
- Py → relay: `H_RLD`, a page reload with a seed (the hard reset).
- Page errors arrive as `H_ERR` and are forwarded, so the env raises immediately.

## Determinism invariants

An episode is a function of its own actions only, for a given Chrome build:

- Physics is fixed-step (`m_world.step(0.04, 5, 5)`). Our steps use
  `TIMESTEP_SIZE = 1/30` for the game clock.
- `CORE.app.host.fixed_delta = TIMESTEP_SIZE`. Without it, `FN_RESET`'s
  `on_internal_update()` advances the clock by wall-clock time.
- `create_player` is wrapped so that each reset first puts the leapfrogging track
  segments back (`FN_RESET_TRACK`) and rebuilds Box2D's broadphase
  (`FN_REBUILD_BROADPHASE`). Otherwise the tree shape left by earlier episodes
  changes the contact order, and with it the trajectory.
- `Math.random` (the seed) never reaches the physics: different seeds give
  identical episodes.
- Guard: `tests/test_live.py::test_episodes_depend_only_on_their_own_actions`. Its
  long-distance gait fails without the broadphase rebuild, and its random probe
  fails without the track reset, so each half of the fix is covered.

## Domain terms

- **Soft reset**: `CMD_RST` → `FN_RESET` → the game's `reset()`. **Hard reset**:
  a page reload (`reload_on_reset`, or `reset(seed=…)`). Both now yield identical
  episodes.
- **Time**: QWOP's displayed seconds, `scoreTime / 10`; one frame is 1/300.
- **Distance**: torso x / 10, in metres. 100 m means success.
- **Step**: `frames_per_step` frames (`stepsize` in the page).
- **Episode end**: the game ends (`terminated`), or the game clock reaches
  `max_episode_time` game-seconds (`truncated`; default 180, `None` for no limit).
- **Observation**: 12 body parts × (x, y, angle, vx, vy), each normalised to
  [−1, 1] over fixed ranges.

## Conventions

- Python ≥ 3.11. Ruff with the maintainer's default config; there is no
  repository-level ruff config.
- The vendored JS is deliberately kept prettier-formatted.
- Validation at the boundaries raises explicitly, never via `assert`, because
  experiments run under `python -O`.

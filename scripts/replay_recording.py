#!/usr/bin/env python3
"""Replay the episodes of a `RecordWrapper` recording at human-understandable speed."""

import argparse
import time

from qwop_gym_leaner.envs.v1.qwop_env import QwopEnv

__all__: list[str] = ["parse_recording", "replay"]

ACTION_NAMES = {
    True: ["NONE", "Q", "W", "O", "P", "QW", "QP", "WO", "OP"],
    False: ["NONE", "Q", "W", "O", "P", "QW", "QO", "QP", "WO", "WP", "OP", "QWO", "QWP", "QOP", "WOP", "QWOP"],
}


def parse_recording(rec_file: str) -> tuple[dict[str, int], list[list[int]]]:
    """
    Parse a recording into its header and its episodes.

    Recording format:
        - Header: `key=value` lines before the first action: `seed`, plus `frames_per_step`
          and `reduced_action_set` in recordings made by this version
        - Then one action integer per line, each episode followed by a `*` line
        - Legacy files may contain `X`-terminated blocks (discarded episodes kept only as
          replay history); they are skipped

    Returns:
        Tuple of (header, episodes) where episodes is a list of action lists

    Raises:
        ValueError: If the file format is invalid
    """
    with open(rec_file, encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    header: dict[str, int] = {}
    episodes: list[list[int]] = []
    current: list[int] = []

    for num, line in enumerate(lines, start=1):
        if not line:
            continue
        if "=" in line:
            if episodes or current:
                raise ValueError(f"line {num}: header line after the first action: {line!r}")
            key, _, value = line.partition("=")
            try:
                header[key] = int(value)
            except ValueError:
                raise ValueError(f"line {num}: invalid header value: {line!r}") from None
        elif line == "*":
            episodes.append(current)
            current = []
        elif line == "X":
            current = []
        else:
            try:
                action = int(line)
            except ValueError:
                raise ValueError(f"line {num}: expected an action, got: {line!r}") from None
            if not 0 <= action <= 15:
                raise ValueError(f"line {num}: action must be between 0 and 15, got: {action}")
            current.append(action)

    if "seed" not in header:
        raise ValueError("missing 'seed=<value>' header line")
    if current:
        raise ValueError(f"{len(current)} trailing actions without a closing '*'")
    return header, episodes


def replay(
    rec_file: str,
    reduced_action_set: bool | None = None,
    frames_per_step: int | None = None,
    delay_per_step: float = 0.1,
    episodes: list[int] | None = None,
):
    """
    Replay recorded episodes.

    Args:
        rec_file: Path to the recording file
        reduced_action_set: Whether to use the reduced action set (default: from the recording)
        frames_per_step: Number of game frames per step (default: from the recording)
        delay_per_step: Delay in seconds between steps (for human-readable speed)
        episodes: 1-based indices of the episodes to replay (all if None)
    """
    header, recorded = parse_recording(rec_file)

    # Legacy recordings have no env parameters in the header: fall back to this script's old defaults
    fps = header.get("frames_per_step", 2) if frames_per_step is None else frames_per_step
    reduced = bool(header.get("reduced_action_set", 1)) if reduced_action_set is None else reduced_action_set
    for key, value in (("frames_per_step", fps), ("reduced_action_set", int(reduced))):
        if key in header and header[key] != value:
            print(f"⚠ Overriding the recorded {key}={header[key]} with {value}: replays will diverge")

    selected = list(range(1, len(recorded) + 1)) if episodes is None else episodes
    for i in selected:
        if not 1 <= i <= len(recorded):
            raise ValueError(f"episode {i} does not exist (the recording has {len(recorded)})")

    names = ACTION_NAMES[reduced]
    invalid = sorted({a for i in selected for a in recorded[i - 1] if a >= len(names)})
    if invalid:
        raise ValueError(f"actions {invalid} are not in the {'reduced' if reduced else 'full'} action set")

    print(f"Recording {rec_file}: seed={header['seed']}, {len(recorded)} episodes")
    print(f"Replaying episodes {selected} with frames_per_step={fps}, reduced_action_set={reduced}\n")

    env = QwopEnv(
        reduced_action_set=reduced,
        frames_per_step=fps,
        loglevel="INFO",
        stat_in_browser=True,
        game_in_browser=True,
        seed=header["seed"],
        max_episode_time=None,  # recorded episodes ran to their own end, however long
    )

    try:
        for i in selected:
            actions = recorded[i - 1]
            _obs, info = env.reset()
            total_reward = 0.0
            terminated = False
            step = 0

            for step, action in enumerate(actions, start=1):
                _obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward

                if step % 10 == 0 or terminated:
                    print(
                        f"Episode {i} | Step {step:4d} | Action: {names[action]:6s} | "
                        f"Reward: {reward:7.3f} | Distance: {info['distance']:6.2f}m | "
                        f"Time: {info['time']:6.2f}s | Total Reward: {total_reward:7.2f}"
                    )

                if terminated or truncated:
                    break
                time.sleep(delay_per_step)

            # Recorded episodes end exactly at their terminal step
            status = "✓" if terminated and step == len(actions) else "✗ diverged from the recording:"
            print(
                f"{status} episode {i} ended after {step}/{len(actions)} steps, "
                f"distance {info['distance']:.2f} m, time {info['time']:.2f} s, reward {total_reward:.2f}\n"
            )
    except KeyboardInterrupt:
        print("\nReplay interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay recorded QWOP episodes at human-understandable speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "rec_file",
        nargs="?",
        default="./random_agent_recording.txt",
        help="Path to the recording file",
    )
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        help="1-based index of an episode to replay (repeatable; default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay in seconds between steps (lower = faster replay)",
    )
    parser.add_argument(
        "--full-actions",
        action="store_true",
        default=None,
        help="Use the full action set (16 actions instead of 9); default: from the recording",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=None,
        help="Number of game frames per step; default: from the recording",
    )

    args = parser.parse_args()

    replay(
        rec_file=args.rec_file,
        episodes=args.episode,
        delay_per_step=args.delay,
        frames_per_step=args.frames_per_step,
        reduced_action_set=None if args.full_actions is None else not args.full_actions,
    )

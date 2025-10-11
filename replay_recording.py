#!/usr/bin/env python3
"""Replay recorded QWOP actions at human-understandable speed."""

import argparse
import time

from qwop_gym_leaner.envs.v1.qwop_env import QwopEnv

__all__: list[str] = ["parse_recording", "replay"]


def parse_recording(rec_file: str) -> tuple[int, list[int]]:
    """
    Parse a recording file and extract seed and actions.

    Returns:
        Tuple of (seed, actions) where actions is a list of action integers
    """
    with open(rec_file, "r") as f:
        lines = f.read().strip().split("\n")

    # First line should be seed=<value>
    if not lines or not lines[0].startswith("seed="):
        raise ValueError(f"Invalid recording file format. First line should be 'seed=<value>'")

    seed = int(lines[0].split("=")[1])
    actions = []

    # Parse actions (stop at * or X which mark episode boundaries)
    for line in lines[1:]:
        line = line.strip()
        if line in ("*", "X", ""):
            break
        actions.append(int(line))

    return seed, actions


def replay(
    rec_file: str,
    reduced_action_set: bool = True,
    frames_per_step: int = 2,
    delay_per_step: float = 0.1,
):
    """
    Replay a recorded episode.

    Args:
        rec_file: Path to the recording file
        reduced_action_set: Whether to use reduced action set (9 vs 16 actions)
        frames_per_step: Number of game frames per step
        delay_per_step: Delay in seconds between steps (for human-readable speed)
    """
    print("\n" + "=" * 70)
    print(" " * 18 + "QWOP Recording Replay")
    print("=" * 70 + "\n")

    # Parse recording
    print(f"Loading recording from: {rec_file}")
    try:
        seed, actions = parse_recording(rec_file)
    except Exception as e:
        print(f"✗ Failed to parse recording file: {e}")
        return

    print(f"✓ Recording loaded successfully!")
    print(f"  Seed: {seed}")
    print(f"  Total actions: {len(actions)}")
    print(f"  Replay speed: {delay_per_step:.3f}s per step")
    print()

    if len(actions) == 0:
        print("⚠ Warning: Recording contains no actions!")
        print("The episode may not have completed successfully during recording.")
        print("Try running the recording script again.")
        return

    # Action names for display
    action_names = (
        ["NONE", "Q", "W", "O", "P", "QW", "QP", "WO", "OP"]
        if reduced_action_set
        else ["NONE", "Q", "W", "O", "P", "QW", "QO", "QP", "WO", "WP", "OP", "QWO", "QWP", "QOP", "WOP", "QWOP"]
    )

    # Create environment
    print("Initializing environment...")
    try:
        env = QwopEnv(
            reduced_action_set=reduced_action_set,
            frames_per_step=frames_per_step,
            loglevel="INFO",
            stat_in_browser=True,
            game_in_browser=True,
            seed=seed,  # Use the same seed as the recording
        )
        print("✓ Environment initialized successfully!")
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        print("\nMake sure 'chrome_for_testing' and 'chromedriver_for_testing'")
        print("are available in your PATH.")
        return

    try:
        print(f"\n{'─' * 70}")
        print("Starting replay...")
        print(f"{'─' * 70}\n")

        # Reset environment
        obs, info = env.reset()
        total_reward = 0.0

        print(f"Starting position: {info['distance']:.2f}m at {info['time']:.2f}s\n")

        # Replay actions
        for step, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Print progress every 10 steps
            if step % 10 == 0 or terminated:
                action_name = action_names[action]
                print(
                    f"Step {step:4d} | Action: {action_name:6s} | "
                    f"Reward: {reward:7.3f} | Distance: {info['distance']:6.2f}m | "
                    f"Time: {info['time']:6.2f}s | Total Reward: {total_reward:7.2f}"
                )

            if terminated or truncated:
                print(f"\nEpisode ended at step {step + 1}")
                break

            # Delay for human-readable speed
            time.sleep(delay_per_step)

        # Final summary
        print(f"\n{'─' * 70}")
        print("Replay Complete")
        print(f"  Steps replayed:  {step + 1}/{len(actions)}")
        print(f"  Total reward:    {total_reward:.2f}")
        print(f"  Distance:        {info['distance']:.2f} meters")
        print(f"  Time elapsed:    {info['time']:.2f} seconds")
        print(f"{'─' * 70}\n")

    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user.")
    finally:
        print("Closing environment...")
        env.close()
        print("✓ Replay complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay recorded QWOP actions at human-understandable speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "rec_file",
        nargs="?",
        default="./random_agent_recording.txt",
        help="Path to the recording file",
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
        help="Use full action set (16 actions instead of 9)",
    )
    parser.add_argument(
        "--frames-per-step",
        type=int,
        default=2,
        help="Number of game frames per step (should match recording)",
    )

    args = parser.parse_args()

    replay(
        rec_file=args.rec_file,
        reduced_action_set=not args.full_actions,
        frames_per_step=args.frames_per_step,
        delay_per_step=args.delay,
    )

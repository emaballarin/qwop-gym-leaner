#!/usr/bin/env python3
# =============================================================================
# Test file for QWOP Gym environment with simulated agent control
# =============================================================================

"""
Test script that simulates agent control in the QWOP environment.
Runs multiple episodes with a random agent and prints detailed progress.
"""

import time
import numpy as np
from qwop_gym.envs.v1.qwop_env import QwopEnv


def print_header():
    """Print a nice header for the test."""
    print("\n" + "=" * 70)
    print(" " * 15 + "QWOP Gym Agent Simulation Test")
    print("=" * 70 + "\n")


def print_episode_header(episode: int, total_episodes: int):
    """Print episode start information."""
    print(f"\n{'─' * 70}")
    print(f"Episode {episode + 1}/{total_episodes}")
    print(f"{'─' * 70}")


def print_step_info(
    step: int,
    action: int,
    reward: float,
    distance: float,
    time_elapsed: float,
    total_reward: float,
    action_names: list[str],
):
    """Print information about the current step."""
    action_name = action_names[action]
    print(
        f"Step {step:4d} | Action: {action_name:6s} | "
        f"Reward: {reward:7.3f} | Distance: {distance:6.2f}m | "
        f"Time: {time_elapsed:6.2f}s | Total Reward: {total_reward:7.2f}"
    )


def print_episode_summary(
    episode: int, steps: int, total_reward: float, distance: float, time_elapsed: float, is_success: bool
):
    """Print episode summary."""
    avg_speed = distance / time_elapsed if time_elapsed > 0 else 0
    status = "SUCCESS! 🏆" if is_success else "Failed ❌"

    print(f"\n{'─' * 70}")
    print(f"Episode {episode + 1} Complete - {status}")
    print(f"  Steps taken:    {steps}")
    print(f"  Total reward:   {total_reward:.2f}")
    print(f"  Distance:       {distance:.2f} meters")
    print(f"  Time elapsed:   {time_elapsed:.2f} seconds")
    print(f"  Average speed:  {avg_speed:.2f} m/s")
    print(f"{'─' * 70}")


def get_action_names(reduced: bool) -> list[str]:
    """Get human-readable action names."""
    if reduced:
        return ["NONE", "Q", "W", "O", "P", "QW", "QP", "WO", "OP"]
    else:
        return ["NONE", "Q", "W", "O", "P", "QW", "QO", "QP", "WO", "WP", "OP", "QWO", "QWP", "QOP", "WOP", "QWOP"]


class RandomAgent:
    """Simple random agent for testing."""

    def __init__(self, action_space, hold_duration: int = 5):
        """
        Initialize random agent.

        Args:
            action_space: The environment's action space
            hold_duration: Number of steps to hold each action (simulates deliberate control)
        """
        self.action_space = action_space
        self.hold_duration = hold_duration
        self.current_action = None
        self.hold_counter = 0

    def select_action(self) -> int:
        """Select an action (holds for multiple steps for realism)."""
        if self.current_action is None or self.hold_counter >= self.hold_duration:
            self.current_action = self.action_space.sample()
            self.hold_counter = 0

        self.hold_counter += 1
        return self.current_action

    def reset(self):
        """Reset the agent's state."""
        self.current_action = None
        self.hold_counter = 0


def run_test(
    num_episodes: int = 3,
    max_steps_per_episode: int = 500,
    print_every: int = 10,
    reduced_action_set: bool = True,
    frames_per_step: int = 2,
):
    """
    Run the agent simulation test.

    Args:
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        print_every: Print progress every N steps
        reduced_action_set: Use reduced action set (9 vs 16 actions)
        frames_per_step: Number of game frames per step (higher = faster simulation)
    """
    print_header()
    print(f"Configuration:")
    print(f"  Episodes:           {num_episodes}")
    print(f"  Max steps/episode:  {max_steps_per_episode}")
    print(f"  Frames per step:    {frames_per_step}")
    print(f"  Reduced actions:    {reduced_action_set}")
    print(f"\nInitializing environment...")

    # Create environment with default chrome_for_testing and chromedriver_for_testing
    try:
        env = QwopEnv(
            reduced_action_set=reduced_action_set,
            frames_per_step=frames_per_step,
            loglevel="INFO",
            stat_in_browser=True,  # Show stats in browser
            game_in_browser=True,  # Show game in browser
        )
        print("✓ Environment initialized successfully!")
        print(f"  Action space size: {env.action_space.n}")
        print(f"  Observation space: {env.observation_space.shape}")
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        print("\nMake sure 'chrome_for_testing' and 'chromedriver_for_testing'")
        print("are available in your PATH, or specify paths explicitly.")
        return

    # Initialize agent
    agent = RandomAgent(env.action_space, hold_duration=5)
    action_names = get_action_names(reduced_action_set)

    # Track overall statistics
    episode_rewards = []
    episode_distances = []
    episode_steps = []

    try:
        # Run episodes
        for episode in range(num_episodes):
            print_episode_header(episode, num_episodes)

            # Reset environment and agent
            obs, info = env.reset()
            agent.reset()

            total_reward = 0.0
            step = 0

            print(f"Starting position: {info['distance']:.2f}m at {info['time']:.2f}s")

            # Run episode
            for step in range(max_steps_per_episode):
                # Select and perform action
                action = agent.select_action()
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward

                # Print progress
                if step % print_every == 0 or terminated:
                    print_step_info(step, action, reward, info["distance"], info["time"], total_reward, action_names)

                # Check if episode ended
                if terminated or truncated:
                    break

                # Small delay to make it visible
                time.sleep(0.05)

            # Print episode summary
            print_episode_summary(episode, step + 1, total_reward, info["distance"], info["time"], info["is_success"])

            # Store statistics
            episode_rewards.append(total_reward)
            episode_distances.append(info["distance"])
            episode_steps.append(step + 1)

            # Brief pause between episodes
            if episode < num_episodes - 1:
                print("\nPreparing next episode...")
                time.sleep(1)

        # Print overall summary
        print("\n" + "=" * 70)
        print(" " * 20 + "Overall Summary")
        print("=" * 70)
        print(f"Episodes completed:     {num_episodes}")
        print(f"Average reward:         {np.mean(episode_rewards):.2f} (±{np.std(episode_rewards):.2f})")
        print(f"Average distance:       {np.mean(episode_distances):.2f}m (±{np.std(episode_distances):.2f})")
        print(f"Average steps:          {np.mean(episode_steps):.1f} (±{np.std(episode_steps):.1f})")
        print(f"Best distance:          {np.max(episode_distances):.2f}m")
        print(f"Best reward:            {np.max(episode_rewards):.2f}")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    finally:
        print("Closing environment...")
        env.close()
        print("✓ Test complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test QWOP Gym environment with a random agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--print-every", type=int, default=10, help="Print progress every N steps")
    parser.add_argument("--full-actions", action="store_true", help="Use full action set (16 actions instead of 9)")
    parser.add_argument(
        "--frames-per-step", type=int, default=2, help="Number of game frames per step (higher = faster)"
    )

    args = parser.parse_args()

    run_test(
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        print_every=args.print_every,
        reduced_action_set=not args.full_actions,
        frames_per_step=args.frames_per_step,
    )

"""
Unit tests for environment.
"""

import gymnasium as gym
import pygame

import gym_collabsort
from gym_collabsort.config import Config
from gym_collabsort.envs.env import CollabSortEnv, RenderMode
from gym_collabsort.envs.robot import Robot


def test_version() -> None:
    """Test environment version"""

    # Check that version string is not empty
    assert gym_collabsort.__version__


def test_render_rgb() -> None:
    """Test env registration and RGB rendering"""

    env = CollabSortEnv(render_mode=RenderMode.RGB_ARRAY)
    env.reset()

    env.step(action=env.action_space.sample())

    frame = env.render()
    assert frame is not None
    assert frame.ndim == 3
    assert frame.shape[0] == env.config.window_dimensions[1]
    assert frame.shape[1] == env.config.board_width


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    env = gym.make("CollabSort-v0")
    env.reset()

    for _ in range(60):
        _, _, _, _, _ = env.step(action=env.action_space.sample())

    env.close()



def test_robotic_agent(pause_at_end: bool = False) -> None:
    """Test an agent using the same behavior as the robot, but with specific rewards"""

    config = Config(n_objects=30)

    env = CollabSortEnv(render_mode=RenderMode.HUMAN, config=config)
    env.reset()

    # Use robot policy with agent rewards
    robotic_agent = Robot(
        board=env.board,
        arm=env.board.agent_arm,
        rewards=config.agent_rewards,
    )

    ep_over: bool = False
    while not ep_over:
        _, _, terminated, truncated, _ = env.step(
            action=robotic_agent.choose_action().value
        )
        ep_over = terminated or truncated

    if pause_at_end:
        # Wait for any user input to exit environment
        pygame.event.clear()
        _ = pygame.event.wait()

    env.close()

def test_robotic_agent_many_episodes(pause_at_end: bool = False) -> None:
    """Test an agent using the same behavior as the robot, but with specific rewards"""

    config = Config(n_objects=30)

    env = CollabSortEnv(render_mode=RenderMode.HUMAN, config=config)
    env.reset()
        
    # Use robot policy with agent rewards
    robotic_agent = Robot(
        board=env.board,
        arm=env.board.agent_arm,
        rewards=config.agent_rewards,
    )

    for i in range(3):
        if i != 0:
            env.reset(reset_board=True)
            print("RESET")

        ep_over: bool = False
        while not ep_over:
            _, _, terminated, truncated, _ = env.step(
                action=robotic_agent.choose_action().value
            )
            ep_over = terminated or truncated
        

    if pause_at_end:
        # Wait for any user input to exit environment
        pygame.event.clear()
        _ = pygame.event.wait()
        
    env.close()


if __name__ == "__main__":
    # Standalone execution with pause at end
    #test_robotic_agent(pause_at_end=True)
    test_robotic_agent_many_episodes(pause_at_end=True)

"""
Unit tests for environment.
"""

import gymnasium as gym
import pygame
import numpy as np

import gym_collabsort
from gym_collabsort.envs.env import CollabSortEnv, RenderMode
from gym_collabsort.envs.robot import Robot, get_color_priorities, get_shape_priorities
from gym_collabsort.model.dqn_agent import DQNAgent, flatten_obs

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

    env = CollabSortEnv(render_mode=RenderMode.HUMAN)
    env.reset()

    # Use robot policy with agent rewards
    robotic_agent = Robot(
        board=env.board,
        arm=env.board.agent_arm,
        color_priorities=get_color_priorities(env.config.agent_color_rewards),
        shape_priorities=get_shape_priorities(env.config.agent_shape_rewards),
    )

    ep_over: bool = False
    i = 0
    while not ep_over:
        #if i > 10: break

        action = robotic_agent.choose_action()
        #action = action=env.action_space.sample()
        print(action)
        _, _, terminated, trucanted, _ = env.step(action=action)
        ep_over = terminated or trucanted

        i += 1

    env.reset()

    ep_over: bool = False
    i = 0
    while not ep_over:
        #if i > 10: break

        action = robotic_agent.choose_action()
        #action = action=env.action_space.sample()
        print(action)
        _, _, terminated, trucanted, _ = env.step(action=action)
        ep_over = terminated or trucanted

        i += 1

    if pause_at_end:
        # Wait for any user input to exit enrironment
        pygame.event.clear()
        _ = pygame.event.wait()

    env.close()






def test_DQN_agent(pause_at_end: bool = False) -> None:
    import time

    start = time.time()
    env = CollabSortEnv(render_mode=RenderMode.NONE)  # or HUMAN to visualize
    obs, _ = env.reset()
    action_size = len(env.board.get_all_objects())
    state = flatten_obs(obs, action_size)

    agent = DQNAgent(
        board=env.board,
        arm=env.board.agent_arm,
        state_size=len(state),
        action_size=action_size,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    ep_over: bool = False

    i = 0
    while not ep_over:

        print("while iter", i)
        action, slot = agent.choose_action(state)

        #print(action, slot)
        next_obs, reward, terminated, truncated, _ = env.step(action=action)
        ep_over = terminated or truncated

        state = flatten_obs(next_obs, action_size)
        i += 1


    if pause_at_end and env.render_mode == RenderMode.HUMAN:
        # Wait for any user input to exit enrironment
        pygame.event.clear()
        _ = pygame.event.wait()

    env.close()
    end = time.time()
    print("DQN test episode time:", end - start)

if __name__ == "__main__":
    # Standalone execution with pause at end
    #test_robotic_agent(pause_at_end=True)
    test_DQN_agent(pause_at_end=True)

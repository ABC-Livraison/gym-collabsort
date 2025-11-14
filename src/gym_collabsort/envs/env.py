"""
Gym environment for a collaborative sorting task.
"""

from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from pygame.math import Vector2

from ..board.arm import Arm
from ..board.board import Board
from ..board.object import Color, Object, Shape
from ..config import Config
from .robot import Robot, get_color_priorities, get_shape_priorities


class RenderMode(StrEnum):
    """Possible render modes for the environment"""

    HUMAN = "human"
    RGB_ARRAY = "rgb_array"
    NONE = "None"


class CollabSortEnv(gym.Env):
    """Gym environment implementing a collaborative sorting task"""

    # Supported render modes
    metadata = {"render_modes": [rm.value for rm in RenderMode]}

    def __init__(
        self,
        render_mode: RenderMode = RenderMode.NONE,
        config: Config | None = None,
    ) -> None:
        """Initialize the environment"""

        if config is None:
            # Use default configuration values
            config = Config()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.config = config

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # Create board
        self.board = Board(config=self.config)

        # Create robot
        self.robot = Robot(
            board=self.board,
            arm=self.board.robot_arm,
            color_priorities=get_color_priorities(config.robot_color_rewards),
            shape_priorities=get_shape_priorities(config.robot_shape_rewards),
        )

        # Define action format: coordinates of target
        self.action_space = self._get_coords_space()

        # Define observation format. See _get_obs() method for details
        self.observation_space = gym.spaces.Dict(
            {
                "self": self._get_coords_space(),
                "objects": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "coords": self._get_coords_space(),
                            # max_length is the maximum number of characters in a color
                            "color": gym.spaces.Text(max_length=10),
                            "shape": gym.spaces.Discrete(n=len(Shape)),
                        }
                    )
                ),
                "robot": self._get_coords_space(),
            }
        )

    def _get_coords_space(self) -> gym.spaces.Space:
        """Helper method to create a Box space for the 2D coordinates of a board element"""

        return gym.spaces.Box(
            low=np.array([0, 0]),
            # Maximum values are bounded by board dimensions
            high=np.array(
                [
                    self.config.board_width,
                    self.config.board_height,
                ]
            ),
            dtype=int,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        # Init the RNG
        super().reset(seed=seed, options=options)

        self.board.reset()
        self.board.populate(rng=self.np_random)

        # Initialize collision tracking
        self._total_collision_count = 0
        self._step_collision_occurred = False
        self._last_dropped_object = None

        # Reset collision penalties on both arms
        self.board.agent_arm.collision_penalty = False
        self.board.robot_arm.collision_penalty = False

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return (self._get_obs(), self._get_info())

    def _get_obs(self) -> dict:
        """
        Return an observation given to the agent.

        An observation is a dictionary containing:
        - the coordinates of agent arm claw
        - the properties of all objects
        - the coordinates of robot arm claw
        """

        objects = tuple(
            self._get_object_props(object=obj) for obj in self.board.objects
        )

        return {
            "self": np.array(self.board.agent_arm.claw.coords),
            "objects": objects,
            "robot": np.array(self.board.robot_arm.claw.coords),
        }

    def _get_info(self) -> dict:
        """Return additional information given to the agent"""

        # Return collision information in the info dict
        return {
            "collision": self._step_collision_occurred,
            "total_collisions": self._total_collision_count,
            "collected": self._last_dropped_object is not None,
            "collision_penalty_agent": self.board.agent_arm.collision_penalty,
            "collision_penalty_robot": self.board.robot_arm.collision_penalty,
        }

    def _get_object_props(self, object: Object) -> dict:
        """Return properties for a specific object"""

        return {
            "coords": np.array(object.coords),
            "color": object.color,
            "shape": object.shape.value,
            "slot": int(object.slot)
        }

    def step(self, action: tuple[int, int]) -> tuple[dict, float, bool, bool, dict]:
        # Reset step collision tracking
        self._step_collision_occurred = False
        self._last_dropped_object = None

        # Store initial collision states
        agent_collision_before = self.board.agent_arm.collision_penalty
        robot_collision_before = self.board.robot_arm.collision_penalty

        # Init reward with a small time penalty
        reward: float = self.config.reward__time_penalty

        # Handle robot action
        robot_dropped_object = self._handle_action(
            arm=self.board.robot_arm,
            action=self.robot.choose_action(),
            other_arm=self.board.agent_arm,
        )

        # Check if robot action caused a collision
        robot_collision_after = self.board.robot_arm.collision_penalty
        if robot_collision_after and not agent_collision_before:
            self._step_collision_occurred = True
            self._total_collision_count += 1

        # Compute robot reward
        reward += (
            self._compute_reward(
                object=robot_dropped_object,
                color_rewards=self.config.robot_color_rewards,
                shape_rewards=self.config.robot_shape_rewards,
            )
            if robot_dropped_object is not None
            else 0
        )

        # Handle agent action
        agent_dropped_object = self._handle_action(
            arm=self.board.agent_arm, action=action, other_arm=self.board.robot_arm
        )

        # Check if agent action caused a collision
        agent_collision_after = self.board.agent_arm.collision_penalty
        if agent_collision_after and not agent_collision_before:
            self._step_collision_occurred = True
            self._total_collision_count += 1

        # Store the last dropped object for info
        self._last_dropped_object = agent_dropped_object

        # Compute agent reward
        reward += (
            self._compute_reward(
                object=agent_dropped_object,
                color_rewards=self.config.agent_color_rewards,
                shape_rewards=self.config.agent_shape_rewards,
            )
            if agent_dropped_object is not None
            else 0
        )

        observation = self._get_obs()
        info = self._get_info()

        # Episode is terminated when all objects have been picked up
        terminated = len(self.board.objects) == 0

        self.board.draw()
        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return observation, reward, terminated, False, info

    def _handle_action(
        self, arm: Arm, action: tuple[int, int], other_arm: Arm
    ) -> Object | None:
        """Handle an action for agent or robot arm"""

        target_coords = Vector2(action[0], action[1])
        dropped_object = arm.move(
            board=self.board, target_coords=target_coords, other_arm=other_arm
        )
        
        return dropped_object

    def _compute_reward(
        self,
        object: Object,
        color_rewards: dict[Color, float],
        shape_rewards: dict[Shape, float],
    ) -> float:
        """Compute the reward for a dropped object"""

        return color_rewards[object.color] + shape_rewards[object.shape]

    def render(self) -> np.ndarray | None:
        if self.render_mode == RenderMode.RGB_ARRAY:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        if self.window is None and self.render_mode == RenderMode.HUMAN:
            # Init pygame display
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(size=self.config.window_dimensions)
            pygame.display.set_caption(self.config.window_title)

        if self.clock is None and self.render_mode == RenderMode.HUMAN:
            self.clock = pygame.time.Clock()

        canvas = self.board.draw()

        if self.render_mode == RenderMode.HUMAN:
            # The following line copies our drawings from canvas to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.config.render_fps)

        else:  # rgb_array
            return self.board.get_frame()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()

"""
Gym environment for a collaborative sorting task.
"""

from enum import StrEnum
from typing import Any

import gymnasium as gym
import numpy as np
import pygame

from ..board.board import Board
from ..board.object import Color, Object, Shape
from ..config import Action, Config
from .robot import Robot


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
        training_mode: bool = False,
    ) -> None:
        """Initialize the environment"""
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.training_mode = training_mode

        if config is None:
            config = Config()
        self.config = config

        self.window: pygame.Surface | None = None
        self.clock = None

        # Create board
        self.board = Board(rng=self.np_random, config=self.config)

        # Create robot with initial rewards
        self.robot = Robot(
            board=self.board,
            arm=self.board.robot_arm,
            rewards=self.config.robot_rewards,
        )

        # Initialize tracking variables
        self.n_removed_objects: int = 0
        self.cumulative_agent_rewards: float = 0
        self.cumulative_robot_rewards: float = 0
        self.total_steps: int = 0

        # Define action format
        self.action_space = gym.spaces.Discrete(len(Action))

        # Define observation format
        self.observation_space = gym.spaces.Dict(
            {
                "self": self._get_coords_space(),
                "objects": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "coords": self._get_coords_space(),
                            "color": gym.spaces.Discrete(n=len(Color)),
                            "shape": gym.spaces.Discrete(n=len(Shape)),
                        }
                    )
                ),
                "robot": self._get_coords_space(),
            }
        )

    def _get_coords_space(self) -> gym.spaces.Space:
        """Helper method to create a Box space for the 2D coordinates (row, col) of a board element"""
        return gym.spaces.Box(
            low=np.array([1, 1]),
            high=np.array([self.config.n_rows, self.config.n_cols]),
        )

    @property
    def collision_penalty(self) -> bool:
        """Return penalty mode status: are arms in penalty mode after a collision?"""
        return (
            self.board.agent_arm.collision_penalty
            or self.board.robot_arm.collision_penalty
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict, dict]:
        """Reset the environment"""
        # Init the RNG
        super().reset(seed=seed, options=options)

        # Reset the entire board
        self.board.reset()
        
        # Reset tracking variables
        self.n_removed_objects = 0
        self.cumulative_agent_rewards = 0
        self.cumulative_robot_rewards = 0
        self.total_steps = 0

        # Reset robot rewards (using current alpha/beta)
        self.robot.rewards = self.config.robot_rewards

        # Add initial object
        self.board.add_object()

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return (self._get_obs(), self._get_info())

    def _get_obs(self) -> dict:
        """Return an observation given to the agent."""
        objects = tuple(obj.get_props() for obj in self.board.objects)

        return {
            "self": self.board.agent_arm.gripper.coords.as_vector(),
            "objects": objects,
            "robot": self.board.robot_arm.gripper.coords.as_vector(),
        }

    def _get_info(self) -> dict:
        """Return additional information given to the agent"""
        return {
            "action_possible": not self.board.agent_arm.moving_back,
            "collision": self.board.agent_arm.collision_penalty or self.board.robot_arm.collision_penalty,
            "collected": self.board.agent_arm.picked_object is not None or self.board.robot_arm.picked_object is not None,
            "agent_reward_total": self.cumulative_agent_rewards,
            "robot_reward_total": self.cumulative_robot_rewards,
            "alpha": self.config.reward_alpha,
            "beta": self.config.reward_beta,
        }

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Execute one time step in the environment"""
        # Init step reward for agent and robot
        agent_reward: float = self.config.step_reward
        robot_reward: float = self.config.step_reward
        
        # Increment total steps
        self.total_steps += 1
        
        # Apply robot action
        robot_action = (
            self.robot.choose_action()
            if not self.robot.arm.moving_back
            else Action.NONE
        )
        robot_collision, robot_placed_object, robot_picked_object = (
            self.board.robot_arm.act(
                action=robot_action,
                objects=self.board.objects,
                other_arm=self.board.agent_arm,
            )
        )

        # Apply agent action
        agent_action = Action(action)
        agent_collision, agent_placed_object, agent_picked_object = (
            self.board.agent_arm.act(
                action=agent_action,
                objects=self.board.objects,
                other_arm=self.board.robot_arm,
            )
        )

        # Calculate movement penalties
        if robot_action in (Action.UP, Action.DOWN):
            robot_reward += self.config.movement_penalty
        if agent_action in (Action.UP, Action.DOWN):
            agent_reward += self.config.movement_penalty

        # Handle collisions
        if robot_collision or agent_collision:
            self.board.robot_arm._picked_object.empty()
            self.board.agent_arm._picked_object.empty()

            agent_reward += self.config.collision_penalty
            robot_reward += self.config.collision_penalty
        else:
            # Handle successful picks/places with alpha/beta scaled rewards
            if robot_placed_object is not None:
                self._move_to_scorebar(object=robot_placed_object, is_agent=False)
                self.n_removed_objects += 1
            elif robot_picked_object is not None:
                # Get scaled reward using current alpha/beta
                robot_reward += robot_picked_object.get_reward(
                    rewards=self.config.robot_rewards
                )

            if agent_placed_object is not None:
                self._move_to_scorebar(object=agent_placed_object, is_agent=True)
                self.n_removed_objects += 1
            elif agent_picked_object is not None:
                # Get scaled reward using current alpha/beta
                agent_reward += agent_picked_object.get_reward(
                    rewards=self.config.agent_rewards
                )

        # Apply none penalty
        if agent_action == Action.NONE:
            agent_reward += self.config.none_penalty

        # Update world state
        self.n_removed_objects += self.board.animate()
        self.cumulative_agent_rewards += agent_reward
        self.cumulative_robot_rewards += robot_reward

        observation = self._get_obs()
        info = self._get_info()

        # Check termination
        terminated = (
            self.n_removed_objects >= self.config.n_objects
            and self.board.agent_arm.picked_object is None
            and self.board.robot_arm.picked_object is None
        )

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return observation, agent_reward, terminated, False, info

    def _move_to_scorebar(self, object: Object, is_agent=True) -> None:
        """Move a placed object to the agent or robot score bar"""
        if is_agent:
            placed_objects = self.board.agent_placed_objects
            y_placed_object = (
                self.board.agent_arm.base.location_abs[1] + self.config.scorebar_height
            )
        else:
            placed_objects = self.board.robot_placed_objects
            y_placed_object = (
                self.board.robot_arm.base.location_abs[1] - self.config.scorebar_height
            )
        x_placed_object = (
            len(placed_objects)
            * (self.config.board_cell_size + self.config.scorebar_margin)
            + self.config.board_cell_size // 2
            + self.config.scorebar_margin
        )

        object.location_abs = (x_placed_object, y_placed_object)
        placed_objects.add(object)

    def render(self) -> np.ndarray | None:
        if self.render_mode == RenderMode.RGB_ARRAY:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        """Render the current state of the environment as a frame"""
        canvas = self.board.draw(
            agent_reward=self.cumulative_agent_rewards,
            robot_reward=self.cumulative_robot_rewards,
            collision_penalty=self.collision_penalty,
        )

        if self.render_mode == RenderMode.HUMAN:
            if self.window is not None:
                font = pygame.font.Font(None, 24)
                scale_text = font.render(f"α={self.config.reward_alpha:.2f}, β={self.config.reward_beta:.2f}", True, (0, 0, 0))
                canvas.blit(scale_text, (self.config.board_width - 200, 10))

        if self.render_mode == RenderMode.HUMAN:
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    size=self.config.window_dimensions
                )
                pygame.display.set_caption(self.config.window_title)

            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.config.render_fps)

        else:
            return self.board.get_frame()

    def close(self) -> None:
        if self.window:
            pygame.display.quit()
            pygame.quit()
            pygame.quit()
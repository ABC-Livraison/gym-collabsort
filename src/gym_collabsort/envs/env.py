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
    ) -> None:
        """Initialize the environment"""

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if config is None:
            # Use default configuration values
            config = Config()

        self.config = config

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window: pygame.Surface | None = None
        self.clock = None

        # Create board
        self.board = Board(rng=self.np_random, config=self.config)

        # Create robot
        self.robot = Robot(
            board=self.board,
            arm=self.board.robot_arm,
            rewards=config.robot_rewards,
        )

        # Number of remaining time steps in penalty mode
        self.remaining_penalty_steps: int = 0

        # Number of removed objects: placed by any arm or fallen from any treadmill.
        # Used to assess the end of episode
        self.n_removed_objects: int = 0

        # Define action format
        self.action_space = gym.spaces.Discrete(len(Action))

        # Define observation format. See _get_obs() method for details
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
            # Coordonates are 1-based
            low=np.array([1, 1]),
            # Maximum values are bounded by board dimensions
            high=np.array(
                [
                    self.config.n_rows,
                    self.config.n_cols,
                ]
            ),
            dtype=int,
        )

    @property
    def collision_penalty(self) -> bool:
        """Return penalty mode status: are arms in penalty mode after a collision?"""

        return (
            self.board.agent_arm.collision_penalty
            or self.board.robot_arm.collision_penalty
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None, reset_board: bool = False
    ) -> tuple[dict, dict]:
        # Init the RNG
        super().reset(seed=seed, options=options)

        # Reset board if requested
        if reset_board:
            self.board.reset()
            self.n_removed_objects = 0

        self.board.add_object()

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return (self._get_obs(), self._get_info())
    
    def _get_obs(self) -> dict:
        """
        Return an observation given to the agent.

        An observation is a dictionary containing:
        - the coordinates of agent arm gripper
        - the properties of all objects
        - the coordinates of robot arm gripper
        """

        objects = tuple(obj.get_props() for obj in self.board.objects)

        return {
            "self": self.board.agent_arm.gripper.coords.as_vector(),
            "objects": objects,
            "robot": self.board.robot_arm.gripper.coords.as_vector(),
        }

    def _get_info(self) -> dict:
        """Return additional information given to the agent"""

        # Arm gripper movement back to its base after an object pickup or a collision is automated.
        # In this case, any action chosen by the agent will not be considered
        return {"action_possible": not self.board.agent_arm.moving_back}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # Init reward
        reward: float = self.config.step_reward

        # Robot can choose an action only if it is not currently moving back to its base
        robot_action = (
            self.robot.choose_action()
            if not self.robot.arm.moving_back
            else Action.NONE
        )
        agent_action = Action(action)
        
        # === IMPROVED ILLEGAL ACTION HANDLING ===
        # Convert illegal actions to safe alternatives instead of NONE
        current_row = self.board.agent_arm.gripper.coords.row
        
        if agent_action == Action.DOWN and current_row >= self.config.n_rows:
            # At bottom, move up instead
            reward += self.config.illegal_move_penalty
            agent_action = Action.UP
        
        elif agent_action == Action.UP and current_row <= 1:
            # At top, move down instead
            reward += self.config.illegal_move_penalty
            agent_action = Action.DOWN
        
        
        
        # Handle robot action
        collision, placed_object = self.board.robot_arm.act(
            action=robot_action,
            objects=self.board.objects,
            other_arm=self.board.agent_arm,
        )
        if collision:
            reward += self.config.collision_reward
        elif placed_object is not None:
            self._move_to_scorebar(object=placed_object, is_agent=False)
            reward += placed_object.get_reward(rewards=self.config.robot_rewards)
            self.n_removed_objects += 1

        # Handle agent action
        collision, placed_object = self.board.agent_arm.act(
            action=agent_action,
            objects=self.board.objects,
            other_arm=self.board.robot_arm,
        )
        
        if collision:
            reward += self.config.collision_reward
        elif placed_object is not None:
            self._move_to_scorebar(object=placed_object, is_agent=True)
            object_reward = placed_object.get_reward(rewards=self.config.agent_rewards)
            reward += object_reward + self.config.successful_pick_bonus
            self.n_removed_objects += 1
        
        # === NEW: DENSE REWARDS FOR GUIDANCE ===
        """if not collision and placed_object is None:  # No collision or successful pick
            # Reward for being on correct row when object passes
            current_row = self.board.agent_arm.gripper.coords.row
            if current_row in [self.config.upper_treadmill_row, self.config.lower_treadmill_row]:
                # Check if any object is on this treadmill
                for obj in self.board.objects:
                    if obj.coords.row == current_row:
                        # Object is on same treadmill row!
                        reward += 0.1  # Small reward for positioning
                        break"""
        
        missed_count = self.board.animate()
        reward += missed_count * self.config.missed_object_penalty
        self.n_removed_objects += missed_count

        observation = self._get_obs()
        info = self._get_info()

        # Episode is terminated when all objects have either been placed by an arm or have fallen from a treadmill
        terminated = (
            self.n_removed_objects >= self.config.n_objects
            and self.board.agent_arm.picked_object is None
            and self.board.robot_arm.picked_object is None
        )

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def _is_valid_pick_location(self, gripper_coords) -> bool:
        """Check if the gripper is at a valid location to pick an object."""
        treadmill_rows = {
            self.config.upper_treadmill_row,
            self.config.lower_treadmill_row
        }
        return gripper_coords.row in treadmill_rows

    def _move_to_scorebar(self, object: Object, is_agent=True) -> None:
        """Move a placed object to the agent or robot score bar"""

        if is_agent:
            placed_objects = self.board.agent_placed_objects
            # Agent score bar is located below the board
            y_placed_object = (
                self.board.agent_arm.base.location_abs[1] + self.config.scorebar_height
            )
        else:
            placed_objects = self.board.robot_placed_objects
            # Robot score bar is located above the board
            y_placed_object = (
                self.board.robot_arm.base.location_abs[1] - self.config.scorebar_height
            )
        x_placed_object = (
            len(placed_objects)
            * (self.config.board_cell_size + self.config.scorebar_margin)
            + self.config.board_cell_size // 2
            + self.config.scorebar_margin
        )

        # Move placed object to appropriate score bar
        object.location_abs = (x_placed_object, y_placed_object)

        # Update placed object list for either agent or robot
        placed_objects.add(object)

    def render(self) -> np.ndarray | None:
        if self.render_mode == RenderMode.RGB_ARRAY:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        """Render the current state of the environment as a frame"""

        canvas = self.board.draw(collision_penalty=self.collision_penalty)

        if self.render_mode == RenderMode.HUMAN:
            if self.window is None:
                # Init pygame display
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    size=self.config.window_dimensions
                )
                pygame.display.set_caption(self.config.window_title)

            if self.clock is None:
                self.clock = pygame.time.Clock()

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
            pygame.quit()

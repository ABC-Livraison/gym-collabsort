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
        training_mode: bool = False,  # Add this parameter
    ) -> None:
        """Initialize the environment"""

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.training_mode = training_mode  # Store training mode flag

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
        # Use appropriate rewards based on training mode
        robot_rewards = self._get_robot_rewards_for_current_mode(0)
        self.robot = Robot(
            board=self.board,
            arm=self.board.robot_arm,
            rewards=robot_rewards,
        )

        # Number of removed objects: placed by any arm or fallen from any treadmill.
        # Used to assess the end of episode
        self.n_removed_objects: int = 0

        # Total rewards for the agent and robot
        self.cumulative_agent_rewards: float = 0
        self.cumulative_robot_rewards: float = 0

        self.total_steps: int = 0

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

    def _get_robot_rewards_for_current_mode(self, step: int) -> np.ndarray:
        """Get robot rewards based on current mode (training or evaluation)"""
        if self.training_mode:
            return self.config.get_robot_rewards_for_step(step)
        else:
            # Evaluation mode: always use final scale
            return self.config.get_robot_rewards_for_step(self.config.exploration_decay_steps)

    def _get_agent_rewards_for_current_mode(self, step: int) -> np.ndarray:
        """Get agent rewards based on current mode (training or evaluation)"""
        if self.training_mode:
            return self.config.get_agent_rewards_for_step(step)
        else:
            # Evaluation mode: always use final scale
            return self.config.get_agent_rewards_for_step(self.config.exploration_decay_steps)

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
        # Init the RNG
        super().reset(seed=seed, options=options)

        # Reset the entire board
        self.board.reset()
        
        self.n_removed_objects = 0
        self.cumulative_agent_rewards = 0
        self.cumulative_robot_rewards = 0

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
    
        return {
            "action_possible": not self.board.agent_arm.moving_back,
            "collision": self.board.agent_arm.collision_penalty or self.board.robot_arm.collision_penalty,
            "collected": self.board.agent_arm.picked_object is not None or self.board.robot_arm.picked_object is not None,
            # Cumulative rewards as displayed in render mode (used by visualizers)
            "agent_reward_total": self.cumulative_agent_rewards,
            "robot_reward_total": self.cumulative_robot_rewards,
        }
        
    # Add this method to your CollabSortEnv class, right after _get_info() method:

    def _get_reward_breakdown(self, agent_reward=0, robot_reward=0, 
                         agent_collision=False, robot_collision=False,
                         agent_action=None, robot_action=None):
        """Calculate and return reward breakdown information (only what's used for plotting)."""
        
        # Calculate penalty this step
        penalty_this_step = 0
        if robot_collision or agent_collision:
            penalty_this_step = self.config.collision_penalty
        if agent_action in (Action.UP, Action.DOWN):
            penalty_this_step += self.config.movement_penalty
        # Robot movement penalty is already included in robot_reward
        
        # Calculate reward difference (agent - robot)
        reward_difference = agent_reward - robot_reward
        
        # Only return what's actually used by visualizers
        return {
            # These are used by RewardBreakdownPlotter
            'agent_reward_this_step': agent_reward,
            'robot_reward_this_step': robot_reward,
            'reward_difference': reward_difference,
            
            # These are used to calculate total_penalty for reward_without_penalty
            'penalty_this_step': penalty_this_step,
        }

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        # Init step reward for agent and robot
        agent_reward: float = self.config.step_reward
        robot_reward: float = self.config.step_reward
        
        # Increment total training steps only in training mode
        if self.training_mode:
            self.total_steps += 1
        
        # Get dynamic rewards based on current mode
        current_agent_rewards = self._get_agent_rewards_for_current_mode(self.total_steps)
        current_robot_rewards = self._get_robot_rewards_for_current_mode(self.total_steps)
        
        # Update robot with current rewards
        self.robot.rewards = current_robot_rewards

        # Apply robot action.
        # Robot can choose an action only if it is not currently moving back to its base
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

        # Compute movement penalties
        if robot_action in (Action.UP, Action.DOWN):
            robot_reward += self.config.movement_penalty
        if agent_action in (Action.UP, Action.DOWN):
            agent_reward += self.config.movement_penalty

        # Handle collisions, placed and picked objects (if any)
        if robot_collision or agent_collision:
            # Immediatly drop any picked object in case of a collision
            self.board.robot_arm._picked_object.empty()
            self.board.agent_arm._picked_object.empty()

            # Compute negative rewards for the collision
            agent_reward += self.config.collision_penalty
            robot_reward += self.config.collision_penalty
        else:
            if robot_placed_object is not None:
                # Robot arm has placed an object: move it to score bar
                self._move_to_scorebar(object=robot_placed_object, is_agent=False)
                # Increment number of objects removed from the board
                self.n_removed_objects += 1
            elif robot_picked_object is not None:
                # Compute robot reward using dynamic rewards
                robot_reward += robot_picked_object.get_reward(
                    rewards=current_robot_rewards  # Use dynamic rewards
                )

            if agent_placed_object is not None:
                # Agent arm has placed an object: move it to score bar
                self._move_to_scorebar(object=agent_placed_object, is_agent=True)
                # Increment number of objects removed from the board
                self.n_removed_objects += 1
            elif agent_picked_object is not None:
                # Compute agent reward using dynamic rewards
                agent_reward += agent_picked_object.get_reward(
                    rewards=current_agent_rewards  # Use dynamic rewards
                )

        # Update world state
        self.n_removed_objects += self.board.animate()
        self.cumulative_agent_rewards += agent_reward
        self.cumulative_robot_rewards += robot_reward

        observation = self._get_obs()
        info = self._get_info()

        # Add reward breakdown using the helper method
        reward_breakdown = self._get_reward_breakdown(
            agent_reward=agent_reward,
            robot_reward=robot_reward,
            agent_collision=agent_collision,
            robot_collision=robot_collision,
            agent_action=agent_action,
            robot_action=robot_action
        )
        
        # Merge reward breakdown into info
        info.update(reward_breakdown)

        # Episode is terminated when all objects have either been placed by an arm or have fallen from a treadmill
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
        canvas = self.board.draw(
            agent_reward=self.cumulative_agent_rewards,
            robot_reward=self.cumulative_robot_rewards,
            collision_penalty=self.collision_penalty,
        )

        # Add scale info in HUMAN mode
        if self.render_mode == RenderMode.HUMAN:
            # Get current scale for info display
            current_scale = self.config.get_current_reward_scale(
                self.total_steps if self.training_mode else self.config.exploration_decay_steps
            )
            
            # Create font if window exists (Pygame is initialized)
            if self.window is not None:
                font = pygame.font.Font(None, 24)
                scale_text = font.render(f"Reward Scale: {current_scale:.2f}x", True, (0, 0, 0))
                canvas.blit(scale_text, (self.config.board_width - 200, 10))
                
                # Add mode indicator
                mode_text = font.render(f"Mode: {'Training' if self.training_mode else 'Evaluation'}", True, (0, 0, 0))
                canvas.blit(mode_text, (self.config.board_width - 200, 40))

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
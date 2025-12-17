"""
Base types and configuration values for gym-collabsort environment.
"""

import math
from dataclasses import dataclass
from typing import Tuple
from enum import Enum

import numpy as np


class Color(Enum):
    """Possible colors for an object"""

    RED = 0
    BLUE = 1
    YELLOW = 2


def get_color_name(color: Color) -> str:
    """Return the name associated to a color"""

    if color == Color.RED:
        return "red"
    elif color == Color.BLUE:
        return "blue"
    elif color == Color.YELLOW:
        return "yellow"


class Shape(Enum):
    """Possible shapes for an object"""

    SQUARE = 0
    CIRCLE = 1
    TRIANGLE = 2


class Action(Enum):
    """Possible actions for agent and robot"""

    # Do nothing or continue a previously initiated movement
    NONE = 0
    # Move the arm gripper up
    UP = 1
    # Move the arm gripper down
    DOWN = 2
    # Pick object at current location
    PICK = 3


@dataclass
class Config:
    """Configuration class with default values"""
    
    # ---------- Reward Scaling Parameters ----------
    # alpha: scale factor for rewards (Theorem 5.1)
    reward_alpha: float = 1.0
    
    # beta: offset for rewards
    reward_beta: float = 0.0

    # Frames Per Second for environment rendering
    render_fps: int = 5

    # ---------- Window and board ----------

    # Number of board rows
    n_rows: int = 10

    # Number of board columns
    n_cols: int = 16

    # Size of a square board cell in pixels
    board_cell_size: int = 50

    @property
    def board_height(self) -> int:
        """Return the height of the board in pixels"""

        return self.n_rows * self.board_cell_size

    @property
    def board_width(self) -> int:
        """Return the width of the board in pixels"""

        return self.n_cols * self.board_cell_size

    # Width in pixels of delimitation line between score bar and board
    scorebar_line_thickness: int = 3

    # Margin around score bar content in pixels
    scorebar_margin: int = 3

    @property
    def scorebar_height(self) -> int:
        """Return the height of the score bar (which is an offset for vertical coordinates)"""

        return self.board_cell_size + self.scorebar_margin

    @property
    def window_dimensions(self) -> tuple[int, int]:
        """Return the dimensions (width, height) of the main window in pixels"""

        # Add heights of scorebars for robot and agent
        return (
            self.board_width,
            self.board_height + self.scorebar_height * 2,
        )

    # Title of the main window
    window_title = "gym-collabsort - Collaborative sorting task"

    # Background color of the window
    background_color: str = "white"

    # ---------- Treadmills ----------

    # Board row for the upper treadmill
    upper_treadmill_row = 4

    # Board row for the lower treadmill
    lower_treadmill_row = 7

    # Thickness of treadmill delimitation lines in pixels
    treadmill_line_thickness: int = 1

    # ---------- Objects ----------

    # Maximum number of objects. If 0, new objects will be added indefinitely
    n_objects: float = math.inf

    # Probability of adding a new object at each time step
    new_object_proba = 0.25

    # ---------- Agent and robot arms ----------

    # Board column where arm bases are placed
    arm_base_col: int = 4

    # Thickness of arm base lines in pixels
    arm_base_line_thickness: int = 5

    # Background color for arm base while in penalty mode
    arm_base_penalty_color: str = "orange"

    # Thickness of the line between arm base and gripper in pixels
    arm_line_thickness: int = 7

    # Size (height & width) of the agent and robot grippers in pixels
    arm_gripper_size: int = board_cell_size // 2

    # ---------- Rewards ----------

    # Base step reward
    step_reward: float = 0

    # Negative reward when a collision happens
    collision_penalty: float = -10

    # Negative reward for movement
    movement_penalty = -1

    # Negative reward for NONE action (Theorem 1.1: makes inaction less attractive)
    none_penalty: float = -0.05

    # Base reward matrices (before alpha/beta scaling)
    @property
    def base_agent_rewards(self) -> np.ndarray:
        """Base rewards for agent (before alpha/beta scaling)"""
        return np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])

    @property
    def base_robot_rewards(self) -> np.ndarray:
        """Base rewards for robot (before alpha/beta scaling)"""
        return np.array([[5, 4, 3], [8, 7, 6], [2, 1, 0]])

    @property
    def agent_rewards(self) -> np.ndarray:
        """Return scaled rewards with alpha and beta applied"""
        base = self.base_agent_rewards
        # Apply scaling: R' = α * R + β
        scaled = self.reward_alpha * base + self.reward_beta
        return scaled

    @property
    def robot_rewards(self) -> np.ndarray:
        """Return scaled rewards with alpha and beta applied"""
        base = self.base_robot_rewards
        # Apply scaling: R' = α * R + β
        scaled = self.reward_alpha * base + self.reward_beta
        return scaled

    # Size in pixels of reward texts
    reward_text_size: int = 16
    
    def copy_with_alpha_beta(self, alpha: float = None, beta: float = None) -> 'Config':
        """Create a copy of the config with new alpha/beta values"""
        import copy
        new_config = copy.deepcopy(self)
        if alpha is not None:
            new_config.reward_alpha = alpha
        if beta is not None:
            new_config.reward_beta = beta
        return new_config
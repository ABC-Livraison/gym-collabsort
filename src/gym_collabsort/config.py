"""
Base types and configuration values.
"""

import math
from dataclasses import dataclass
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

    # Board row for the uppoer treadmill
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
    
    # ========== Exploration parameters ==========
    initial_exploration_temp: float = 1.0      # Start: 10:1 ratio
    final_exploration_temp: float = 0.3        # End: 3:1 ratio  
    exploration_decay_steps: int = 100000
    
    # ========== Rewards ==========
    # Base step reward
    step_reward: float = 0

    # Negative reward when a collision happens
    collision_penalty: float = -10

    # Negative reward for movement
    movement_penalty = -1

    # Original pick-up rewards
    _agent_raw_rewards = np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])
    _robot_raw_rewards = np.array([[5, 4, 3], [8, 7, 6], [2, 1, 0]])
    
    BASE_REWARD_SCALE: float = 10.0  # Renamed for clarity
    
    sum_rewards = abs(movement_penalty) + abs(collision_penalty)
    movement_penalty = (movement_penalty/sum_rewards)
    collision_penalty = (collision_penalty/sum_rewards)
    
    def get_exploration_factor(self, step: int) -> float:
        """Get current exploration factor (like temperature)"""
        if step >= self.exploration_decay_steps:
            return self.final_exploration_temp
        
        decay = step / self.exploration_decay_steps
        return (self.initial_exploration_temp * (1 - decay) + 
                self.final_exploration_temp * decay)
    
    def get_current_reward_scale(self, step: int) -> float:
        """Get reward scale for current training step"""
        return self.BASE_REWARD_SCALE * self.get_exploration_factor(step)
    
    def get_agent_rewards_for_step(self, step: int) -> np.ndarray:
        """Get agent rewards scaled for current training step"""
        original_rewards = np.array([[8, 7, 6], [5, 4, 3], [2, 1, 0]])
        sum_abs_rewards = np.sum(np.abs(original_rewards))
        current_scale = self.get_current_reward_scale(step)
        return (original_rewards / sum_abs_rewards) * current_scale
    
    def get_robot_rewards_for_step(self, step: int) -> np.ndarray:
        """Get robot rewards scaled for current training step"""
        original_rewards = np.array([[5, 4, 3], [8, 7, 6], [2, 1, 0]])
        sum_abs_rewards = np.sum(np.abs(original_rewards))
        current_scale = self.get_current_reward_scale(step)
        return (original_rewards / sum_abs_rewards) * current_scale
    
    @property
    def agent_rewards(self) -> np.ndarray:
        """Base agent rewards (used when no step info available)"""
        return self.get_agent_rewards_for_step(0)  # Default to full scale
    
    @property
    def robot_rewards(self) -> np.ndarray:
        """Base robot rewards (used when no step info available)"""
        return self.get_robot_rewards_for_step(0)  # Default to full scale
    
    # Size in pixels of reward texts
    reward_text_size: int = 16
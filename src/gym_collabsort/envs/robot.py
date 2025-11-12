"""
Implementation of robot policy.
"""

from ..board.arm import Arm
from ..board.board import Board
from ..config import Color, Shape


def get_color_priorities(color_rewards: dict[Color, float]) -> list[Color]:
    """Return the ordered list of color priorities based on rewards"""

    # Sort colors by descending reward
    return list(
        color
        for color, _ in sorted(
            color_rewards.items(), key=lambda item: item[1], reverse=True
        )
    )


def get_shape_priorities(shape_rewards: dict[Shape, float]) -> list[Shape]:
    """Return the ordered list of shape priorities based on rewards"""

    # Sort shapes by descending reward
    return list(
        shape
        for shape, _ in sorted(
            shape_rewards.items(), key=lambda item: item[1], reverse=True
        )
    )


class Robot:
    def __init__(
        self,
        board: Board,
        arm: Arm,
        color_priorities: tuple[Color],
        shape_priorities: tuple[Shape],
    ) -> None:
        self.board = board
        self.arm = arm
        self.color_priorities = color_priorities
        self.shape_priorities = shape_priorities

        # Coordinates of current target (an object or the arm base)
        self.target_coords: tuple[int, int] = None

    def choose_action(self) -> tuple[int, int]:
        """Return the coordinates of the chosen target"""

        if self.arm.is_retracted():
            # Reset target when arm is fully retracted
            self.target_coords = None

        elif self.arm.collision_penalty or self.arm.picked_object is not None:
            # Retract arm towards its base after a collision or if a object has been picked
            self.target_coords = self.arm.base.coords
            return self.target_coords
        
        elif (
            self.target_coords is not None
            and self.board.get_object_at(self.target_coords) is None
        ):
            # Previously targeted object is no longer there (probably picked by the other arm).
            # Retract arm towards its base
            self.target_coords = self.arm.base.coords
            return self.target_coords
        
        if self.target_coords is None:
            # Search for objects compatible with picking priorities
            compatible_objects = self.board.get_compatible_objects(
                colors=self.color_priorities,
                shapes=self.shape_priorities,
            )
            if len(compatible_objects) > 0:
                # Aim for the first compatible object
                self.target_coords = compatible_objects[0].coords

        if self.target_coords is not None:
            # Move arm towards target
            return self.target_coords
        else:
            # No possible target => stay still
            return self.arm.claw.coords

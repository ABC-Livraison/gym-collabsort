"""
The environment board and its content.
"""

import numpy as np
import pygame
from pygame.math import Vector2
from pygame.sprite import Group

from ..config import Color, Config, Shape
from .arm import Arm
from .object import Object


class Board:
    """The environment board"""

    def __init__(self, rng: np.random.Generator, config: Config | None = None) -> None:
        if config is None:
            # Use default configuration values
            config = Config()

        self.rng = rng
        self.config = config

        # Define the surface to draw upon
        self.canvas = pygame.Surface(size=self.config.window_dimensions)

        # Create an empty group for objects
        self.objects: Group[Object] = Group()

        # Number of added objects since the beginning of the episode
        self.n_added_objects: int = 0

        # Create agent and robot arms
        self.agent_arm = Arm(
            location=Vector2(
                x=(self.config.arm_base_col - 0.5) * self.config.board_cell_size,
                y=self.config.board_height - self.config.board_cell_size // 2,
            ),
            config=config,
        )
        self.robot_arm = Arm(
            location=Vector2(
                x=(self.config.arm_base_col - 0.5) * self.config.board_cell_size,
                y=self.config.board_cell_size // 2,
            ),
            config=config,
        )

        self.agent_placed_objects: Group[Object] = Group()
        self.robot_placed_objects: Group[Object] = Group()


    def reset(self) -> None:
        """Reset the board to its initial state"""
        
        # Reset canvas
        self.canvas = pygame.Surface(size=self.config.window_dimensions)

        # Reset counter
        self.n_added_objects = 0

        # Reset objects
        self.objects = Group()
        self.agent_placed_objects = Group()
        self.robot_placed_objects = Group()

        # Reset agent arm
        self.agent_arm._picked_object.empty()
        self.agent_arm.collision_penalty = False
        self.agent_arm.gripper.location = self.agent_arm.base.location.copy()

        # Reset robot arm
        self.robot_arm._picked_object.empty()
        self.robot_arm.collision_penalty = False
        self.robot_arm.gripper.location = self.robot_arm.base.location.copy()



    def add_object(
        self,
    ) -> None:
        """Add a new object to the board"""

        # Randomly choose object treadmill
        if self.rng.choice((0, 1)):
            obj_y: float = (
                self.config.upper_treadmill_row - 0.5
            ) * self.config.board_cell_size
        else:
            obj_y: float = (
                self.config.lower_treadmill_row - 0.5
            ) * self.config.board_cell_size

        # Randomly generate object attributes
        obj_color: Color = self.rng.choice(Color)
        obj_shape: Shape = self.rng.choice(Shape)

        new_obj = Object(
            location=Vector2(
                x=round(self.config.board_cell_size * (self.config.n_cols - 0.5)),
                y=obj_y,
            ),
            color=obj_color,
            shape=obj_shape,
            config=self.config,
        )
        self.objects.add(new_obj)
        self.n_added_objects += 1

    def animate(self) -> int:
        """
        Animate the board: move existing objects and possibly add a new one.
        Return the number of fallen objects
        """

        # Number of object which have fallen from any treadmill after this movement
        n_fallen_objects: int = 0

        # Only non-picked objects move on treadmills
        moving_objects = [
            obj
            for obj in self.objects
            if (
                obj != self.robot_arm.picked_object
                and obj != self.agent_arm.picked_object
            )
        ]

        # Move all non-picked objects from right to left on their treadmill
        for obj in moving_objects:
            obj.move(col_offset=-1)

            if obj.location[0] < 0:
                # Object has fallen from the treadmill before being picked
                self.objects.remove(obj)

                n_fallen_objects += 1

        # Add a new object according to probability if limite has not been reached yet
        if (
            self.n_added_objects < self.config.n_objects
            and self.rng.random() < self.config.new_object_proba
        ):
            self.add_object()

        return n_fallen_objects

    def draw(self, collision_penalty: bool = False) -> pygame.Surface:
        """Draw the board"""

        # fill the surface with background color to wipe away anything previously drawed
        self.canvas.fill(self.config.background_color)

        # Draw board limits.
        for y in (0, self.config.board_height):
            pygame.draw.line(
                surface=self.canvas,
                color="black",
                # Y-axis (vertical) values are offsetted to take into account the robot score bar above the board
                start_pos=(0, y + self.config.scorebar_height),
                end_pos=(
                    self.config.board_width,
                    y + self.config.scorebar_height,
                ),
                width=self.config.scorebar_line_thickness,
            )

        # Draw placed objects for each arm
        self.agent_placed_objects.draw(surface=self.canvas)
        self.robot_placed_objects.draw(surface=self.canvas)

        # Draw bases for each arm
        self.agent_arm.base.update_image(collision_penalty=collision_penalty)
        self.agent_arm._base.draw(surface=self.canvas)
        self.robot_arm.base.update_image(collision_penalty=collision_penalty)
        self.robot_arm._base.draw(surface=self.canvas)

        # Draw objects
        self.objects.draw(surface=self.canvas)

        # Draw treadmills lines just aboce and below objects
        for treadmill_row in (
            self.config.upper_treadmill_row - 1,
            self.config.upper_treadmill_row,
            self.config.lower_treadmill_row - 1,
            self.config.lower_treadmill_row,
        ):
            pygame.draw.line(
                surface=self.canvas,
                color="black",
                # Y-axis (vertical) values are offsetted to take into account the robot score bar above the board
                start_pos=(
                    0,
                    treadmill_row * self.config.board_cell_size
                    + self.config.scorebar_height,
                ),
                end_pos=(
                    self.config.board_width,
                    treadmill_row * self.config.board_cell_size
                    + self.config.scorebar_height,
                ),
                width=self.config.treadmill_line_thickness,
            )

        # Draw picked objects (if any)
        if self.agent_arm.picked_object is not None:
            self.agent_arm._picked_object.draw(surface=self.canvas)
        if self.robot_arm.picked_object is not None:
            self.robot_arm._picked_object.draw(surface=self.canvas)

        # Draw agent arm gripper
        self.agent_arm._gripper.draw(surface=self.canvas)
        # Draw line between agent arm base and gripper
        pygame.draw.line(
            surface=self.canvas,
            color="black",
            start_pos=self.agent_arm.base.location_abs,
            end_pos=self.agent_arm.gripper.location_abs,
            width=self.config.arm_line_thickness,
        )

        # Draw robot arm gripper
        self.robot_arm._gripper.draw(surface=self.canvas)
        # Draw line between robot arm base and gripper
        pygame.draw.line(
            surface=self.canvas,
            color="black",
            start_pos=self.robot_arm.base.location_abs,
            end_pos=self.robot_arm.gripper.location_abs,
            width=self.config.arm_line_thickness,
        )

        return self.canvas

    def get_frame(self) -> np.ndarray:
        """Return the board as a NumPy array"""

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
        )

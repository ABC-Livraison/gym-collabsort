"""
The environment board and its content.
"""

import numpy as np
import pygame
from pygame.math import Vector2
from pygame.sprite import Group, spritecollide

from ..config import Color, Config, Shape
from .arm import Arm
from .object import Object


class Board:
    """The environment board"""

    def __init__(self, config: Config | None = None) -> None:
        if config is None:
            # Use default configuration values
            config = Config()

        self.config = config

        # Define the surface to draw upon
        self.canvas = pygame.Surface(size=self.config.window_dimensions)

        # Create an empty group for objects
        self.objects: Group[Object] = Group()

        # Create agent and robot arms
        self.agent_arm = Arm(
            coords=Vector2(
                x=self.config.board_width // 2,
                y=self.config.board_height - self.config.arm_base_size // 2,
            ),
            config=config,
        )
        self.robot_arm = Arm(
            coords=Vector2(
                x=self.config.board_width // 2,
                y=self.config.arm_base_size // 2,
            ),
            config=config,
        )

        self.agent_dropped_objects: Group[Object] = Group()
        self.robot_dropped_objects: Group[Object] = Group()
        self._next_slot = 0


    def reset(self):
        self.canvas = pygame.Surface(size=self.config.window_dimensions)
        self.objects: Group[Object] = Group()

        # self.agent_arm = Arm(
        #     coords=Vector2(
        #         x=self.config.board_width // 2,
        #         y=self.config.board_height - self.config.arm_base_size // 2,
        #     ),
        #     config=self.config,
        # )
        # self.robot_arm = Arm(
        #     coords=Vector2(
        #         x=self.config.board_width // 2,
        #         y=self.config.arm_base_size // 2,
        #     ),
        #     config=self.config,
        # )

        self._next_slot = 0
        self.agent_dropped_objects: Group[Object] = Group()
        self.robot_dropped_objects: Group[Object] = Group()


    def populate(
        self,
        rng: np.random.Generator,
    ) -> None:
        """Populate the board"""

        # Add objects to the board in an available location
        remaining_objects = self.config.n_objects
        while remaining_objects > 0:
            # Randoml generate coordinates compatible with board dimensions
            obj_coords = Vector2(
                x=rng.integers(
                    low=self.config.object_size // 2,
                    high=self.config.board_width - self.config.object_size // 2,
                ),
                y=rng.integers(
                    low=self.config.object_size // 2,
                    high=self.config.board_height - self.config.object_size // 2,
                ),
            )
            # Randomly generate object attributes
            obj_color = rng.choice(a=self.config.object_colors)
            obj_shape = rng.choice(a=self.config.object_shapes)

            new_obj = Object(
                coords=obj_coords,
                color=obj_color,
                shape=obj_shape,
                config=self.config,
                slot=self._next_slot
            )
            if (
                not self.agent_arm.collide_sprite(sprite=new_obj)
                and not self.robot_arm.collide_sprite(sprite=new_obj)
                and not spritecollide(sprite=new_obj, group=self.objects, dokill=False)
            ):
                # Add new object if it doesn't collide with anything already present on the board
                self.objects.add(new_obj)
                self._next_slot += 1
                remaining_objects -= 1

    def get_object_at(self, coords: tuple[int, int]) -> Object | None:
        """Return the object at a given location, if any"""

        for obj in self.objects:
            if obj.coords == coords:
                return obj

    def get_all_objects(self):
        available_objects = [
            obj
            for obj in self.objects
            if obj != self.agent_arm.picked_object
            and obj != self.robot_arm.picked_object
        ]

        return available_objects

    def get_compatible_objects(
        self, colors: tuple[Color], shapes: tuple[Shape]
    ) -> list[Object]:
        """
        Get the ordered list of board objects with listed colors and shapes.

        Desired colors and shapes are given by descending order of priority.
        Selected objects (if any) are returned by descending order or compatibility.
        Color is used as first selection criterion, shape as second.
        """

        shape_compatible_objects: list[Object] = []
        compatible_objects: list[Object] = []

        # Exclude already picked objects
        available_objects = [
            obj
            for obj in self.objects
            if obj != self.agent_arm.picked_object
            and obj != self.robot_arm.picked_object
        ]

        # Select available object that are shape-compatible.
        # They are sorted by descending order of shape priority
        for shape in shapes:
            for obj in available_objects:
                if obj.shape == shape:
                    shape_compatible_objects.append(obj)

        # Select shape-compatible objects that are also color-compatible.
        # They are sorted by descending order of color priority
        for color in colors:
            for obj in shape_compatible_objects:
                if obj.color == color:
                    compatible_objects.append(obj)

        return compatible_objects

    def draw(self) -> pygame.Surface:
        """Draw the board"""

        # fill the surface with background color to wipe away anything previously drawed
        self.canvas.fill(self.config.background_color)

        # Draw board limits.
        # Y is offsetted to take into account the dropped objects line above the board
        for y in (0, self.config.board_height):
            pygame.draw.line(
                surface=self.canvas,
                color="black",
                start_pos=(0, y + self.config.y_offset),
                end_pos=(self.config.board_width, y + self.config.y_offset),
                width=self.config.board_line_width,
            )

        # An object just dropped by the agent arm must be moved below the board
        if self.agent_arm._dropped_object:
            # Move dropped object to line above the board
            self.agent_arm.dropped_object.coords_abs = (
                len(self.agent_dropped_objects)
                * (self.config.object_size + self.config.dropped_object_margin)
                + self.config.object_size // 2
                + self.config.dropped_object_margin,
                self.agent_arm.base.coords_abs[1] + self.config.y_offset,
            )
            # Update objects lists
            self.agent_dropped_objects.add(self.agent_arm.dropped_object)
            self.objects.remove(self.agent_arm.dropped_object)
            self.agent_arm._dropped_object.empty()

        # An object just dropped by the robot arm must be moved above the board
        if self.robot_arm._dropped_object:
            # Move dropped object to line below the board
            self.robot_arm.dropped_object.coords_abs = (
                len(self.robot_dropped_objects)
                * (self.config.object_size + self.config.dropped_object_margin)
                + self.config.object_size // 2
                + self.config.dropped_object_margin,
                self.robot_arm.base.coords_abs[1] - self.config.y_offset,
            )
            # Update objects lists
            self.robot_dropped_objects.add(self.robot_arm.dropped_object)
            self.objects.remove(self.robot_arm.dropped_object)
            self.robot_arm._dropped_object.empty()

        # Draw dropped objects for each arm
        self.agent_dropped_objects.draw(surface=self.canvas)
        self.robot_dropped_objects.draw(surface=self.canvas)

        # Draw bases for each arm
        self.agent_arm._base.draw(surface=self.canvas)
        self.robot_arm._base.draw(surface=self.canvas)

        # Draw objects
        self.objects.draw(surface=self.canvas)

        # Draw agent arm claw
        self.agent_arm._claw.draw(surface=self.canvas)
        # Draw line between agent arm base and claw
        pygame.draw.line(
            surface=self.canvas,
            color="black",
            start_pos=self.agent_arm.base.coords_abs,
            end_pos=self.agent_arm.claw.coords_abs,
            width=self.config.arm_line_width,
        )

        # Draw robot arm claw
        self.robot_arm._claw.draw(surface=self.canvas)
        # Draw line between robot arm base and claw
        pygame.draw.line(
            surface=self.canvas,
            color="black",
            start_pos=self.robot_arm.base.coords_abs,
            end_pos=self.robot_arm.claw.coords_abs,
            width=self.config.arm_line_width,
        )

        return self.canvas

    def get_frame(self) -> np.ndarray:
        """Return the board as a NumPy array"""

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
        )

    def get_object_by_slot(self, slot: int) -> Object | None:
        for obj in self.objects:
            if getattr(obj, "slot", None) == slot:
                return obj
        # si l'objet a été droppé (retiré), il ne sera plus dans self.objects
        return None

    def get_slot_by_coords(self, coords: tuple[int,int]) -> int | None:
        for obj in self.objects:
            if tuple(obj.coords) == tuple(coords):
                return obj.slot
        return None

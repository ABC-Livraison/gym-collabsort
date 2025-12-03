"""
Arm-related definitions.
"""

from __future__ import annotations

import pygame
from pygame.math import Vector2
from pygame.sprite import Group, GroupSingle, spritecollide

from ..config import Action, Config
from .object import Object
from .sprite import Sprite


class Base(Sprite):
    """Base of the agent or robot arm"""

    def __init__(self, location: Vector2, config: Config) -> None:
        super().__init__(
            location=location,
            size=config.board_cell_size,
            config=config,
        )

    def update_image(self, collision_penalty: bool = False) -> None:
        """Update the image of the arm base before drawing it"""

        # Set color based on collision penalty state
        color = (
            self.config.background_color
            if not collision_penalty
            else self.config.arm_base_penalty_color
        )
        self.image.fill(color=color)

        # Draw an empty square box
        # Draw vertical lines
        for x in (0, self.config.board_cell_size - 1):
            pygame.draw.line(
                surface=self.image,
                color="black",
                start_pos=(x, 0),
                end_pos=(x, self.config.board_cell_size),
                width=self.config.arm_base_line_thickness,
            )
        # Draw horizontal lines
        for y in (0, self.config.board_cell_size - 1):
            pygame.draw.line(
                surface=self.image,
                color="black",
                start_pos=(0, y),
                end_pos=(self.config.board_cell_size, y),
                width=self.config.arm_base_line_thickness,
            )


class Gripper(Sprite):
    """Gripper of the agent or robot arm"""

    def __init__(self, location: Vector2, config: Config) -> None:
        super().__init__(
            location=location,
            size=config.arm_gripper_size,
            config=config,
            transparent_background=True,
        )

        self.config = config

        pygame.draw.circle(
            surface=self.image,
            color="black",
            center=(config.arm_gripper_size // 2, config.arm_gripper_size // 2),
            radius=config.arm_gripper_size // 2,
        )


class Arm:
    def __init__(self, location: Vector2, config: Config) -> None:
        self.config = config

        # Create arm base
        self._base: GroupSingle[Base] = GroupSingle(
            Base(location=location, config=self.config)
        )

        # Create arm gripper
        self._gripper: GroupSingle[Gripper] = GroupSingle(
            Gripper(location=location, config=self.config)
        )

        # Create empty single sprite group for picked object
        self._picked_object: GroupSingle[Object] = GroupSingle()

        # Collision penalty status: True after a collision and while gripper hasn't come back to its base
        self.collision_penalty: bool = False

    @property
    def base(self) -> Base:
        """Return the arm base"""

        return self._base.sprite

    @property
    def gripper(self) -> Gripper:
        """Return the arm gripper"""

        return self._gripper.sprite

    @property
    def picked_object(self) -> Object | None:
        """Return the picked object (if any)"""

        return self._picked_object.sprite

    @property
    def moving_back(self) -> bool:
        """Return True if the arm is moving back to its base after an object pickup or a collision"""

        return self.picked_object is not None or self.collision_penalty

    def collide_arm(self, arm: Arm) -> bool:
        """Check if the arm collides with the other arm"""

        # Only grippers can collide
        return (
            len(spritecollide(sprite=arm.gripper, group=self._gripper, dokill=False))
            > 0
        )

    def act(
        self,
        action: Action,
        objects: Group[Object],
        other_arm: Arm,
    ) -> tuple[bool, Object | None, Object | None]:
        """
        Handle the chosen action for the arm. Return:
        - collision status
        - the placed object if movement ends in arm base with a picked object
        - the picked object if arm gripper successfully picks an object
        """

        # Defaults for returned values
        collision: bool = False
        placed_object: Object | None = None
        picked_object: Object | None = None

        if self.moving_back:
            # Move back arm gripper to its base automatically
            if self.gripper.coords.row > self.base.coords.row:
                # Moving back the robot arm gripper to its base
                collision, placed_object = self._move(
                    row_offset=-1, objects=objects, other_arm=other_arm
                )
            elif self.gripper.coords.row < self.base.coords.row:
                # Moving back the agent arm gripper to its base
                collision, placed_object = self._move(
                    row_offset=1, objects=objects, other_arm=other_arm
                )

        if action == Action.UP:
            # Move the gripper up (no possible object placement)
            collision, _ = self._move(
                row_offset=-1, objects=objects, other_arm=other_arm
            )

        elif action == Action.DOWN:
            # Move the gripper down (no possible object placement)
            collision, _ = self._move(
                row_offset=1, objects=objects, other_arm=other_arm
            )

        elif action == Action.PICK:
            # Only check for a pickable object if no object has already been picked
            if self.picked_object is None:
                # Check if the gripper can pick an object at current location
                pickable_objects = [
                    obj for obj in objects if obj.location == self.gripper.location
                ]
                # Only one object may be at the same location as the arm gripper
                if len(pickable_objects) == 1:
                    # Pick object at current location
                    picked_object = pickable_objects[0]
                    self._picked_object.add(picked_object)

        return collision, placed_object, picked_object

    def _move(
        self, row_offset: int, objects: Group[Object], other_arm: Arm
    ) -> tuple[bool, Object | None]:
        """
        Move the arm gripper vertically (up if row offset is negative, down otherwise).
        Return collision status and the placed object if movement ends in arm base with a picked object
        """

        collision: bool = False
        placed_object: Object | None = None

        # Move arm gripper
        self.gripper.move(row_offset=row_offset)

        if self.picked_object is not None:
            # Move the picked object alongside gripper
            self.picked_object.move(row_offset=row_offset)

        if self.collide_arm(arm=other_arm):
            collision = True

            # Change collision penalty status (both grippers will move back to their base)
            self.collision_penalty = True
            other_arm.collision_penalty = True

            if other_arm.picked_object is not None:
                # Drop any object picked by the other arm just before this collision
                other_arm._picked_object.empty()

        if self.is_retracted():
            # Gripper is back to its base: cancel collision penalty (if any)
            self.collision_penalty = False

            if self.picked_object is not None:
                # The placed object will be returned
                placed_object = self.picked_object

                # Arm has finished moving the object to its base
                self._picked_object.remove(placed_object)
                objects.remove(placed_object)

        return collision, placed_object

    def is_retracted(self) -> bool:
        """Check if the arm is entirely retracted (gripper has returned to base)"""

        return self.gripper.location == self.base.location
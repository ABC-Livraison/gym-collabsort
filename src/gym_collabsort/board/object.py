""" """

import pygame
from pygame.math import Vector2

from ..config import Color, Config, Shape
from .sprite import Sprite


class Object(Sprite):
    """A pickable object"""

    def __init__(
        self,
        coords: Vector2,
        config: Config,
        color: Color,
        shape: Shape,
        slot: int
    ) -> None:
        super().__init__(
            coords=coords,
            size=config.object_size,
            config=config,
            transparent_background=True,
        )

        self.color = color
        self.shape = shape
        self.slot = int(slot)

        # Draw object on the image
        if self.shape == Shape.SQUARE:
            self.image.fill(color=color)
        elif self.shape == Shape.CIRCLE:
            pygame.draw.circle(
                surface=self.image,
                color=self.color,
                center=(config.object_size // 2, config.object_size // 2),
                radius=config.object_size // 2,
            )
        elif self.shape == Shape.TRIANGLE:
            # Compute coordinates of 3 vectices
            top = (config.object_size // 2, 0)
            bl = (0, config.object_size)
            br = (config.object_size, config.object_size)

            # Draw the triangle
            pygame.draw.polygon(
                surface=self.image, color=self.color, points=(top, bl, br)
            )

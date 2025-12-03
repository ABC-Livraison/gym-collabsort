""" """

import numpy as np
import pygame
from pygame.math import Vector2

from ..config import Color, Config, Shape, get_color_name
from .sprite import Sprite


class Object(Sprite):
    """A pickable object"""

    def __init__(
        self,
        location: Vector2,
        config: Config,
        color: Color,
        shape: Shape,
    ) -> None:
        super().__init__(
            location=location,
            size=config.board_cell_size,
            config=config,
            transparent_background=True,
        )

        self.color = color
        self.shape = shape

        color_name = get_color_name(color=color)

        # Draw object on the image
        if self.shape == Shape.SQUARE:
            self.image.fill(color=color_name)
        elif self.shape == Shape.CIRCLE:
            pygame.draw.circle(
                surface=self.image,
                color=color_name,
                center=(config.board_cell_size // 2, config.board_cell_size // 2),
                radius=config.board_cell_size // 2,
            )
        elif self.shape == Shape.TRIANGLE:
            # Compute coordinates of 3 vectices
            top = (config.board_cell_size // 2, 0)
            bl = (0, config.board_cell_size)
            br = (config.board_cell_size, config.board_cell_size)

            # Draw the triangle
            pygame.draw.polygon(
                surface=self.image, color=color_name, points=(top, bl, br)
            )

    def get_reward(self, rewards: np.ndarray) -> float:
        """Return the reward associated to this object"""

        return float(rewards[self.color.value, self.shape.value])

    def get_props(self) -> dict:
        """Return object properties"""

        return {
            "coords": self.coords.as_vector(),
            "color": self.color.value,
            "shape": self.shape.value,
        }

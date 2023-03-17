import numpy as np
import numpy.typing as npt

from .Die import Die
from enum import Enum

from utils.common import CellType, TrapType
from utils.constants import STARTING_CELL, SLOW_LANE_FIRST_CELL, SLOW_LANE_LAST_CELL, FAST_LANE_FIRST_CELL, FAST_LANE_LAST_CELL

class BoardGame:
	def __init__(self, layout: npt.NDArray, dice: list[Die], circle: bool = False) -> None:
		self.layout = layout
		self.layout_size = len(layout)
		self.final_cell = self.layout_size - 1
		self.dice = dice
		self.circle = circle
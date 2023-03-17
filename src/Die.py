import numpy as np

from enum import Enum

class DieType(Enum):
	SECURITY = 1
	NORMAL = 2
	RISKY = 3

class Die:
	def __init__(self, type: DieType, moves: list[int], trap_triggering_probability: float) -> None:
		self.type = type.name
		self.moves = moves
		self.trap_triggering_probability = trap_triggering_probability

	def roll(self):
		return np.random.choice(self.moves)
	
	def is_triggering_trap(self) -> int:
		return np.random.choice([0, 1], p=[1 - self.trap_triggering_probability, self.trap_triggering_probability])
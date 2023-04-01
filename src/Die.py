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

	def roll(self) -> int:
		"""Roll the die.

		Returns:
			int: the number returned by the die.
		"""
		return np.random.choice(self.moves)
	
	def is_triggering_trap(self) -> bool:
		"""check if we trigger the trap (if any) based on the probability of the die to trigger a trap.

		Returns:
			bool: True or False depending if we trigger the trap or not.
		"""
		return np.random.choice([True, False], p=[self.trap_triggering_probability, 1 - self.trap_triggering_probability])
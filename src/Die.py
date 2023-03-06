import numpy as np

from enum import Enum

class DieType(Enum):
	SECURITY = 1
	NORMAL = 2
	RISKY = 3

class Die:
	def __init__(self, type: DieType, moves: list[int], trap_triggering_probability: float) -> None:
		self.type = type
		self.moves = moves
		self.trap_triggering_probability = trap_triggering_probability
	
	def roll(self) -> tuple[int, int]:
		next_move = np.random.choice(self.moves)
		is_triggering_trap = np.random.choice([0, 1], p=[1 - self.trap_triggering_probability, self.trap_triggering_probability])

		return (next_move, is_triggering_trap)
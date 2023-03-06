import numpy as np
import numpy.typing as npt

from Die import Die
from enum import Enum

from utils.constants import STARTING_CELL, SLOW_LANE_FIRST_CELL, SLOW_LANE_LAST_CELL, FAST_LANE_FIRST_CELL, FAST_LANE_LAST_CELL

class TrapType(Enum):
	NONE = 0
	RESTART = 1
	PENALTY = 2
	PRISON = 3
	GAMBLE = 4

class CellType(Enum):
	STARTING_CELL = 0
	SLOW_LANE_FIRST_CELL = 3
	SLOW_LANE_LAST_CELL = 9
	FAST_LANE_FIRST_CELL = 10
	FAST_LANE_LAST_CELL = 13

class BoardGame:
	def __init__(self, layout: npt.NDArray, dice: list[Die], circle: bool = False) -> None:
		self.layout = layout
		self.layout_size = len(layout)
		self.final_cell = self.layout_size - 1
		self.dice = dice
		self.circle = circle

		transition_matrix = np.zeros((len(self.layout), len(self.layout)), dtype=float)	
	
	def update_transition_matrix(self, transition_matrix: npt.NDArray, die: Die):
		"""compute the transition matrix for each possible moves allowed by a given die

		Args:
			transition_matrix (npt.NDArray): _description_
			die (Die): _description_
		"""
		# for each state (which corresponds to each possible cell)
		for initial_cell in range(0, len(self.layout)):
			# for each possible move
			for move in die.moves:
				for destination_cell, probability in self.make_move(
					initial_cell=initial_cell,
					amount=move
				):
					transition_matrix[initial_cell, destination_cell] = probability

					# check if we are triggering next cell trap
					if die.is_triggering_trap():
						trap_cell = self.manage_trap(cell=destination_cell)
						transition_matrix[destination_cell, trap_cell] = 1.0

	def make_move(self, initial_cell: int, amount: int, probability: float = 1.0) -> list[tuple[int, float]]:
		"""compute the next cell the agent will move to along with the probability to jump to this cell. 

		Args:
			initial_cell (int): correspond to the array index(which is "cell number - 1") of the agent position and therefore a number in [0..14].
			amount (int): amount by which the agent will move, that is, a number in [0...3] depending on the dice type (security, normal or risky).
			probability (float): probability to move to the destination cell.

		Returns:
			List[Tuple[next_cell: int, proba: float]]: a list of tuples, each one containing the next cell and the probability to jump into it.
		"""
		# possibility to switch to a slow lane
		if initial_cell == 2:
			if amount == 1:
				# go slow or fast lane with probability=0.5
				return [(SLOW_LANE_FIRST_CELL, 0.5), (FAST_LANE_FIRST_CELL, 0.5)]
			else:
				# continue on fast lane with probability=1.0
				return [(initial_cell + amount, probability)]
		
		elif initial_cell in [7, 8, 9]:
			destination_cell = self.manage_slow_lane_special_cases(initial_cell=initial_cell, amount=amount)
			return self.move_toward_final_cell(destination_cell=destination_cell, probability=probability)
			
		
		elif initial_cell in [11, 12, 13]:
			destination_cell = initial_cell + amount
			return self.move_toward_final_cell(destination_cell=destination_cell, probability=probability)

		# already on final cell, do not move agent
		elif initial_cell == self.final_cell:
			return [(initial_cell, probability)]
		
		else:
			return [((initial_cell + amount), probability)]

	def manage_slow_lane_special_cases(self, initial_cell: int, amount: int) -> int:
		if initial_cell + amount > SLOW_LANE_LAST_CELL:
			cells_left = amount - (SLOW_LANE_LAST_CELL - initial_cell)
			return 13 + cells_left

		else:
			return initial_cell + amount
	
	def move_toward_final_cell(self, destination_cell: int, probability: float = 1.0) -> list[tuple[int, float]]:
		if self.circle:
			if destination_cell > self.final_cell:
				# go back to starting point
				return [(STARTING_CELL, probability)]
			else:
				# move as usual
				return [(destination_cell, probability)]
		else:
			# ensure win if overtake the final cell
			return [(min(self.final_cell, destination_cell), probability)]

	def manage_trap(self, cell: int):
		if cell < 0 or cell > 14:
			print("cell index should be in the range [0..14]")
			return

		# check the trap (i.e. reward)
		trap_type = self.layout[cell]

		if trap_type == TrapType.RESTART:
			# teleport back to 1st square (restart)
			return CellType.STARTING_CELL
		elif trap_type == TrapType.PENALTY:
			# teleport 3 steps backward (penalty)

			# fast lane case
			if cell >= 10 and cell <= 12:
				return cell - 7 - 3

			return max(0, cell - 3)
		elif trap_type == TrapType.PRISON:
			# wait one turn before playing again (prison)
			pass
		elif trap_type == TrapType.GAMBLE:
			# randomly teleport anywhere on the board (gamble)
			return np.random.randint(0, 15)
		else:
			# no trap
			return cell
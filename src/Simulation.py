import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from .BoardGame import BoardGame
from .Die import Die, DieType

from utils.common import TrapType, StrategyType
from utils.constants import FAST_LANE_FIRST_CELL, NUMBER_OF_SIMULATIONS, STARTING_CELL, SLOW_LANE_FIRST_CELL, SLOW_LANE_LAST_CELL

class Simulation(BoardGame):
	def __init__(self, layout: npt.NDArray, dice: list[Die], circle: bool = False) -> None:
		super().__init__(layout, dice, circle)
	
	def simulate(self, best_dice: npt.NDArray, strategy: StrategyType, verbose=False):
		layout_size = len(self.layout)
		empirical_costs = np.zeros(layout_size)

		for cell in tqdm(range(0, layout_size)):
			total_cost = 0.0
			for _ in range(0, NUMBER_OF_SIMULATIONS):
				cost = 0.0
				current_cell = cell

				while (current_cell < self.final_cell):
					optimal_die = best_dice[current_cell]
					die = self.dice[self.get_die_index(strategy=strategy, optimal_die=optimal_die)]
					move = die.roll()

					destination_cell = self.get_destination_cell(initial_cell=current_cell, amount=move)
					destination_cell, in_jail = self.manage_trap(destination_cell=destination_cell, die=die)
					next_cost = (2.0 if in_jail else 1.0)
					
					if verbose:
						print("==================")
						print(f"optimal_die: {optimal_die} -- random move: {move}")
						print(f"initial cell: {current_cell} -- destination_cell: {destination_cell} -- in jail: {in_jail}")
						print(f"cost: {cost}")
						print("==================")
				
					cost += next_cost
					current_cell = destination_cell
				
				total_cost += cost
			
			mean_cost = total_cost / NUMBER_OF_SIMULATIONS
			empirical_costs[cell] = mean_cost
		
		return empirical_costs
	
	def get_die_index(self, strategy: StrategyType, optimal_die: int):
		if strategy == StrategyType.OPTIMAL:
			return optimal_die

		elif strategy == StrategyType.SECURITY:
			return 0
		elif strategy == StrategyType.NORMAL:
			return 1
		elif strategy == StrategyType.RISKY:
			return 2

		elif strategy == StrategyType.RANDOM:
			return np.random.randint(0, 3)
			
		elif strategy == StrategyType.SECURITY_NORMAL:
			return np.random.choice([0, 1])
		elif strategy == StrategyType.SECURITY_RISKY:
			return np.random.choice([0, 2])
		elif strategy == StrategyType.NORMAL_RISKY:
			return np.random.choice([1, 2])
		elif strategy == StrategyType.SECURITY_OPTIMAL:
			return np.random.choice([0, optimal_die])
		elif strategy == StrategyType.NORMAL_OPTIMAL:
			return np.random.choice([1, optimal_die])
		elif strategy == StrategyType.RISKY_OPTIMAL:
			return np.random.choice([2, optimal_die])
		else:
			print("unimplemented strategy...")
			raise ValueError()

	def get_destination_cell(self, initial_cell: int, amount: int) -> int:
		"""compute the next cell the agent will move to. 

		Args:
			initial_cell (int): correspond to the array index(which is "cell number - 1") of the agent position and therefore a number in [0..14].
			amount (int): amount by which the agent will move, that is, a number in [0...3] depending on the dice type (security, normal or risky).
			probability (float): probability to move to the destination cell.

		Returns:
			int: the next cell the agent will move to.
		"""
		# possibility to switch to a slow lane
		if initial_cell == 2:
			if amount == 1:
				go_fast_lane = np.random.choice([0, 1])
				if (go_fast_lane):
					return FAST_LANE_FIRST_CELL
				else:
					return SLOW_LANE_FIRST_CELL
			else:
				# continue on fast lane 
				return initial_cell + amount
		
		elif initial_cell in [7, 8, 9]:
			destination_cell = self.manage_slow_lane_special_cases(initial_cell=initial_cell, amount=amount)
			return self.move_toward_final_cell(destination_cell=destination_cell)
			
		
		elif initial_cell in [11, 12, 13]:
			destination_cell = initial_cell + amount
			return self.move_toward_final_cell(destination_cell=destination_cell)

		# already on final cell, do not move agent
		elif initial_cell == self.final_cell:
			return initial_cell
		
		else:
			return initial_cell + amount

	def manage_slow_lane_special_cases(self, initial_cell: int, amount: int) -> int:
		if initial_cell + amount > SLOW_LANE_LAST_CELL:
			cells_left = amount - (SLOW_LANE_LAST_CELL - initial_cell)
			return 13 + cells_left
		else:
			return initial_cell + amount
	
	def move_toward_final_cell(self, destination_cell: int) -> int:
		if self.circle:
			if destination_cell > self.final_cell:
				# go back to starting point
				return STARTING_CELL
			else:
				# move as usual
				return destination_cell
		else:
			# ensure win if overtake the final cell
			return min(self.final_cell, destination_cell)
	
	def manage_trap(self, destination_cell: int, die: Die):
		if destination_cell < 0 or destination_cell > 14:
			print("destination cell should be in the range [0..14]")
			raise ValueError()

		# check the trap (i.e. reward)
		destination_trap_type = int(self.layout[destination_cell])

		if destination_trap_type == TrapType.NONE.value:
			# no trap
			return destination_cell, False
		else:
			# fall onto a trap
			if die.is_triggering_trap():
				return self.trigger_trap(destination_cell=destination_cell, trap_type=destination_trap_type)
			else:
				return destination_cell, False

	def trigger_trap(self, destination_cell: int, trap_type: int):
		if trap_type == TrapType.RESTART.value:
			# teleport back to 1st square (restart)
			return STARTING_CELL, False

		elif trap_type == TrapType.PENALTY.value:
			# teleport 3 steps backward (penalty)
			# fast lane case
			if destination_cell >= 10 and destination_cell <= 12:
				return destination_cell - 7 - 3, False
			else:
				return max(0, destination_cell - 3), False

		elif trap_type == TrapType.PRISON.value:
			# wait one turn before playing again (prison) (= extra_cost)
			return destination_cell, True

		elif trap_type == TrapType.GAMBLE:
			# randomly teleport anywhere on the board (gamble)
			return np.random.randint(0, 15), True

		else:
			# no trap
			return destination_cell, False
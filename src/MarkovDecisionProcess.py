import numpy as np
import numpy.typing as npt

from .Die import Die

from utils.common import CellType, TrapType
from utils.constants import INITIAL_DELTA, EPSILON, STARTING_CELL, SLOW_LANE_FIRST_CELL, SLOW_LANE_LAST_CELL, FAST_LANE_FIRST_CELL, FAST_LANE_LAST_CELL

from .BoardGame import BoardGame

class MarkovDecisionProcess(BoardGame):
	def __init__(self, layout: npt.NDArray, dice: list[Die], circle: bool = False) -> None:
		super().__init__(layout, dice, circle)

	def launch_iteration_value(self) -> list[npt.NDArray]:
		"""_summary_

		Returns:
			list[npt.NDArray]: _description_
		"""
		# expected cost associated to the 14 squares of the game (excluding the goal square)
		# we start from the final state, setting all the values to 0
		# this is V(s)
		Expec = np.zeros(self.layout_size)
		# choice of the best dice for each of the 14 squares (excluding the goal square)
		Dice = np.ones(self.layout_size, dtype=int)

		delta = INITIAL_DELTA

		while delta > EPSILON:
			# V(s') = V(s)
			V_prev = Expec.copy()
			
			# "quality matrix" which contains for each possible action, the cost for each state
			quality_matrix = np.zeros((len(self.dice), self.layout_size))
			#print(quality_matrix)

			# for each cell (i.e. each state)
			# we compute the Bellman optimality conditions V(s)
			for state in range(0, self.layout_size):
				# we consider each dice (i.e each strategy/policy/action)
				# and retrieve the minimum
				for (idx, action) in enumerate(self.dice):
					# c(a|s)
					cost = self.get_cost(die=action, cell=state)
					# V = c(a|s) + \sum_{all states s'} (P(s'|s,a) * V(s')) 
					# 	= c(a|s) + ( P(S'|s,a) \cdot V(S') )
					V = cost + np.dot(self.get_transition_matrix(die=action)[state], V_prev)

					quality_matrix[idx, state] = V
				
				# get the index of the optimal conditions: V(s) (i.e. get the best dice type for each cell)
				dice = np.argmin(quality_matrix[:, state])
				# update the array of best dices
				Dice[state] = dice
				# update the array of costs
				Expec[state] = quality_matrix[dice, state]
			
			# check if we converged toward epsilon
			delta = np.max(np.abs(Expec - V_prev))
			print(delta)
		
		return [Expec[:-1], Dice[:-1]]
	
	def compute_transition_matrices(self):
		self.transition_matrices = {}

		for die in self.dice:
			self.transition_matrices[die.type] = np.zeros((len(self.layout), len(self.layout)))
			self.update_transition_matrix(die=die)
	
	def update_transition_matrix(self, die: Die):
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
					amount=move,
					probability=1/len(die.moves)
				):
					self.compute_probabilities(die=die, initial_cell=initial_cell, destination_cell=destination_cell, probability=probability)

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
				# go slow or fast lane with halved probability
				return [(SLOW_LANE_FIRST_CELL, probability / 2), (FAST_LANE_FIRST_CELL, probability / 2)]
			else:
				# continue on fast lane 
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

	def compute_probabilities(self, die: Die, initial_cell: int, destination_cell: int, probability: float) -> None:
		"""_summary_

		Args:
			die (Die): _description_
			initial_cell (int): _description_
			destination_cell (int): _description_
			probability (float): _description_

		Raises:
			ValueError: _description_
		"""
		if destination_cell < 0 or destination_cell > 14:
			print("destination cell should be in the range [0..14]")
			raise ValueError()

		# check the trap (i.e. reward)
		destination_trap_type = int(self.layout[destination_cell])

		if destination_trap_type == TrapType.NONE.value:
			self.transition_matrices[die.type][initial_cell, destination_cell] += probability
		elif destination_trap_type == TrapType.RESTART.value:
			self.transition_matrices[die.type][initial_cell, destination_cell] += probability * (1 - die.trap_triggering_probability)
			# teleport back to 1st square (restart)
			self.transition_matrices[die.type][destination_cell, STARTING_CELL] += probability * die.trap_triggering_probability
		elif destination_trap_type == TrapType.PENALTY.value:
			self.transition_matrices[die.type][initial_cell, destination_cell] += probability * (1 - die.trap_triggering_probability)
			# teleport 3 steps backward (penalty)
			# fast lane case
			if destination_cell >= 10 and destination_cell <= 12:
				self.transition_matrices[die.type][destination_cell, destination_cell - 7 - 3] += probability * die.trap_triggering_probability
			else:
				self.transition_matrices[die.type][destination_cell, max(0, destination_cell - 3)] += probability * die.trap_triggering_probability
		elif destination_trap_type == TrapType.PRISON.value:
			# wait one turn before playing again (prison) (= extra_cost, see get_cost() method)
			self.transition_matrices[die.type][initial_cell, destination_cell] += probability
		elif destination_trap_type == TrapType.GAMBLE:
			self.transition_matrices[die.type][initial_cell, destination_cell] += probability * (1 - die.trap_triggering_probability)
			# randomly teleport anywhere on the board (gamble)
			self.transition_matrices[die.type][destination_cell, np.random.randint(0, 15)] += probability * die.trap_triggering_probability

	def get_cost(self, die: Die, cell: int) -> float:
		"""compute the cost of an action given a state.
		Each time we throw a dice (and make a potential move), we get +1 cost. 
		We can get an extra cost if we could potentially trigger a trap and go the the jail.

		Returns:
			float: cost for an action (= a die type) given a state (= a cell type): c(a|s)
		"""

		# starting from a given cell, we need to get the cells indices where the jails are located
		jail_indices = np.where(self.layout == TrapType.PRISON.value)[0]
		# we multiply each probability to move to a jail cell by the probability of triggering trap
		# and sum all these costs
		extra_cost = np.sum(self.get_transition_matrix(die)[cell][jail_indices] * die.trap_triggering_probability)

		return 1.0 + extra_cost

	def get_transition_matrix(self, die: Die) -> npt.NDArray:
		return self.transition_matrices[die.type]
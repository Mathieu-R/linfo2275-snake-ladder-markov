import numpy as np
import numpy.typing as npt

from .Die import Die, DieType

from utils.common import CellType, TrapType
from utils.constants import INITIAL_DELTA, EPSILON, MAX_ITER, STARTING_CELL, SLOW_LANE_FIRST_CELL, SLOW_LANE_LAST_CELL, FAST_LANE_FIRST_CELL, FAST_LANE_LAST_CELL

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
		Expec = np.ones(self.layout_size)
		Expec[self.layout_size - 1] = 0.0 # V(d) = 0
		# choice of the best dice for each of the 14 squares (excluding the goal square)
		Dice = np.ones(self.layout_size, dtype=int)

		delta = INITIAL_DELTA
		iterations = 1

		while delta > EPSILON:
			# V(s') = V(s)
			V_prev = Expec.copy()
			#print(V_prev)

			# for each cell (i.e. each state)
			# we compute the Bellman optimality conditions V(s)
			for state in range(0, self.layout_size):

				# for each die, we keep track of the expected cost 
				# of to reach the final state from the current state
				current_state_costs = np.zeros((len(self.dice)))

				# we consider each die (i.e each action)
				for (idx, action) in enumerate(self.dice):
					V = self.compute_bellman_optimal_value(
						possible_moves=self.get_adjacent_matrix(die=action)[state],
						prev=V_prev
					)
					current_state_costs[idx] = V
				
				# get the index of the optimal conditions: V(s) (i.e. get the best die type for each cell)
				die = np.argmin(current_state_costs)
				# update the array of best dices
				Dice[state] = die
				# update the array of costs
				Expec[state] = current_state_costs[die]
			
			Expec[self.layout_size - 1] = 0.0
			
			# check if we converged toward epsilon
			#print(np.subtract(Expec, V_prev))
			delta = np.max(np.abs(np.subtract(Expec, V_prev)))
			iterations += 1
		
		return [Expec[:-1], Dice[:-1]]

	def compute_bellman_optimal_value(self, possible_moves: npt.NDArray, prev: npt.NDArray):
		V = 0.0

		# V = c(a|s) + \sum_{all states s'} (P(s'|s,a) * V(s')) 
		for (next_state, probability, cost) in possible_moves:
			V += probability * (cost + prev[next_state])

		return V
	
	def compute_transition_matrices(self):
		self.adjacent_matrices = {}

		for die in self.dice:
			# adjacent matrix is more convenient than a transition matrix
			# for each state, we compute the possible next states reachables
			# in a list of tuples (next_state, probability, cost)
			self.adjacent_matrices[die.type] = self.init_adjacent_matrix()
			self.update_adjacent_matrix(die=die)

	def init_adjacent_matrix(self):
		adjacent_matrix = []
		for _ in range(0, self.layout_size):
			adjacent_matrix.append([])
		return adjacent_matrix
	
	def update_adjacent_matrix(self, die: Die):
		"""compute the transition matrix for each possible moves allowed by a given die

		Args:
			transition_matrix (npt.NDArray): _description_
			die (Die): _description_
		"""
		# for each state (which corresponds to each possible cell)
		for initial_cell in range(0, len(self.layout)):
			# for each possible move
			for move in die.moves:
				# get the next state along with its probability to move into
				for destination_cell, probability in self.make_move(
					initial_cell=initial_cell,
					amount=move,
					probability=1/len(die.moves)
				):
					# compute the cost and probability to move into the next state
					# (taking trap into account)
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
		# we do not move
		if amount == 0:
			return [(initial_cell + amount, probability)]

		# we arrived on switching cell, possibility to switch to a slow lane or a fast lane
		if initial_cell == 2:
			# go slow or fast lane with halved probability
			return [(SLOW_LANE_FIRST_CELL + (amount - 1), probability / 2), (FAST_LANE_FIRST_CELL + (amount - 1), probability / 2)]
		
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

		move_and_trap_triggered_prob = probability * die.trap_triggering_probability
		move_and_trap_not_triggered_prob = probability * (1 - die.trap_triggering_probability)

		# we did not move into a trap or we used the security dice
		if destination_trap_type == TrapType.NONE.value or die.type == DieType.SECURITY.name:
			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, probability, 1))
		
		elif destination_trap_type == TrapType.RESTART.value:
			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, move_and_trap_not_triggered_prob, 1))
			# teleport back to 1st square (restart)
			self.adjacent_matrices[die.type][initial_cell].append((STARTING_CELL, move_and_trap_triggered_prob, 1))
			
		elif destination_trap_type == TrapType.PENALTY.value:
			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, move_and_trap_not_triggered_prob, 1))
			self.adjacent_matrices[die.type][initial_cell].append((self.teleport_3_step_backward(destination_cell=destination_cell), move_and_trap_triggered_prob, 1))
			
		elif destination_trap_type == TrapType.PRISON.value:
			# wait one turn before playing again (prison) (= extra cost if triggering jail trap)
			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, move_and_trap_triggered_prob, 2))

			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, move_and_trap_not_triggered_prob, 1))

		elif destination_trap_type == TrapType.GAMBLE.value:
			self.adjacent_matrices[die.type][initial_cell].append((destination_cell, move_and_trap_not_triggered_prob, 1))
			# randomly teleport anywhere on the board (with uniform probability)
			for cell in range(0, self.layout_size):
				self.adjacent_matrices[die.type][initial_cell].append((cell, (1 / self.layout_size) * move_and_trap_not_triggered_prob, 1))

	def teleport_3_step_backward(self, destination_cell: int):
		# teleport 3 steps backward (penalty)
		# fast lane case
		if destination_cell >= 10 and destination_cell <= 12:
			return destination_cell - 7 - 3
		else:
			return max(0, destination_cell - 3)

	def get_adjacent_matrix(self, die: Die) -> npt.NDArray:
		return self.adjacent_matrices[die.type]
from sys import last_traceback
import numpy as np
import numpy.typing as npt

from typing import List, Tuple
from src.BoardGame import BoardGame
from src.Die import Die, DieType

from utils.constants import GAMMA, INITIAL_DELTA, EPSILON

def markovDecision(layout: npt.NDArray, circle: bool = False) -> list[npt.NDArray]:
	"""launch the markov decision algorithm process to determine optimal strategy regarding 
	the choice of the dice in the snake and ladder games using the "value iteration" method.

	Args:
		layout (npt.NDArray): represents the layout of the game, each index represent a square and each value represent 
		circle (bool): indicate if the player must land exactly on the final square (circle = true) or still win 
			by overstepping the final square (circle = false)

	Returns:
		list[npt.NDArray]: a list containing two vectors as numpy arrays: Expec and Dice
	"""
	boardGame = BoardGame(
		layout=layout, 
		dice=[
			Die(type=DieType.SECURITY, moves=[0, 1], trap_triggering_probability=0.0),
			Die(type=DieType.NORMAL, moves=[0, 1, 2], trap_triggering_probability=0.5),
			Die(type=DieType.RISKY, moves=[0, 1, 2, 3], trap_triggering_probability=1.0)
		], 
		circle=circle
	)

	boardGame.compute_transition_matrices()

	layout_size = len(layout)

	# expected cost associated to the 14 squares of the game (excluding the goal square)
	# we start from the final state, setting all the values to 0
	# this is V(s)
	Expec = np.zeros(layout_size - 1)
	# choice of the best dice for each of the 14 squares (excluding the goal square)
	Dice = np.ones(layout_size - 1, dtype=int)

	delta = INITIAL_DELTA


	while delta < EPSILON:
		# V(s') = V(s)
		last_bellman_optimality_conditions = Expec.copy()
		# for each cell (i.e. each state ?)
		# we compute the Bellman optimality conditions V(s)
		for cell in range(0, layout_size):
			# we consider each dice (i.e each strategegy/policy/action)
			# and retrieve the minimum
			actions_set = []
			for die in boardGame.dice:
				# c(a|s)
				cost = boardGame.get_cost(die=die, cell=cell)
				# c(a|s) + \sum_{all states s'} (P(s'|s,a) * V(s')) 
				# = c(a|s) + (P(S'|s,a) \cdot V(S'))
				bellman_value = cost + np.dot(boardGame.get_transition_matrix(die=die)[cell], last_bellman_optimality_conditions)

				actions_set.append(bellman_value)
			
			# get the index of the optimal conditions: V(s) (i.e. get the best dice type for each cell)
			dice_type = np.argmin(actions_set)
			# update the array of best dices
			Dice[cell] = dice_type
			# update the array of costs
			Expec[cell] = actions_set[dice_type]
		
		# check if we converged toward epsilon
		delta = max(Expec - last_bellman_optimality_conditions)

	return [Expec, Dice]

def generate_layout():
	layout = np.zeros((15))
	for i in range(1, 14):
		# generate a cell of type between 0 and 4
		# first and last cell are excluded
		layout[i] = np.random.randint(0, 5)
	return layout
	
if __name__ == "__main__":
	#layout = np.array([0, 1, 2, 3, 1, 0, 0, 2, 1, 3, 0, 4, 0, 3, 0])
	layout = generate_layout()
	markovDecision(layout=layout, circle=True)
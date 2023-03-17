import numpy as np
import numpy.typing as npt

from typing import List, Tuple
from src.MarkovDecisionProcess import MarkovDecisionProcess
from src.Simulation import Simulation
from src.Die import Die, DieType

from utils.common import DICE, StrategyType
from utils.constants import INITIAL_DELTA, EPSILON, NUMBER_OF_SIMULATIONS

def markovDecision(layout: npt.NDArray, circle: bool = False) -> list[npt.NDArray]:
	"""launch the markov decision algorithm process to determine optimal strategy regarding 
	the choice of the dice in the snake and ladder games using the "value iteration" method.
	notation: state = a cell ([0..14]) ; action = a die ([SECURITY, NORMAL, RISKY])

	Args:
		layout (npt.NDArray): represents the layout of the game, each index represents a cell (i.e. a state) and each value represents a trap type
		circle (bool): indicate if the player must land exactly on the final square (circle = true) or still win 
			by overstepping the final square (circle = false)

	Returns:
		list[npt.NDArray]: a list containing two vectors as numpy arrays: Expec and Dice
	"""
	mdp = MarkovDecisionProcess(
		layout=layout, 
		dice=DICE, 
		circle=circle
	)

	mdp.compute_transition_matrices()
	layout_size = len(layout)

	# expected cost associated to the 14 squares of the game (excluding the goal square)
	# we start from the final state, setting all the values to 0
	# this is V(s)
	Expec = np.zeros(layout_size)
	# choice of the best dice for each of the 14 squares (excluding the goal square)
	Dice = np.ones(layout_size, dtype=int)

	delta = INITIAL_DELTA

	while delta > EPSILON:
		# V(s') = V(s)
		V_prev = Expec.copy()
		
		# "quality matrix" which contains for each possible action, the cost for each state
		quality_matrix = np.zeros((len(mdp.dice), layout_size))
		#print(quality_matrix)

		# for each cell (i.e. each state ?)
		# we compute the Bellman optimality conditions V(s)
		for state in range(0, layout_size):
			# we consider each dice (i.e each strategegy/policy/action)
			# and retrieve the minimum
			for (idx, action) in enumerate(mdp.dice):
				# c(a|s)
				cost = mdp.get_cost(die=action, cell=state)
				# c(a|s) + \sum_{all states s'} (P(s'|s,a) * V(s')) 
				# = c(a|s) + (P(S'|s,a) \cdot V(S'))
				V = cost + np.dot(mdp.get_transition_matrix(die=action)[state], V_prev)

				quality_matrix[idx] = V
				print(quality_matrix)
			
			# get the index of the optimal conditions: V(s) (i.e. get the best dice type for each cell)
			dice_type = np.argmin(quality_matrix)
			#print(dice_type)
			# update the array of best dices
			Dice[state] = dice_type
			# update the array of costs
			Expec[state] = quality_matrix[dice_type, state]
		
		# check if we converged toward epsilon
		delta = np.max(np.abs(Expec - V_prev))

	return [Expec[:-1], Dice[:-1]]

def generate_layout():
	layout = np.zeros((15))
	for i in range(1, 14):
		# generate a cell of type between 0 and 4
		# first and last cell are excluded
		layout[i] = np.random.randint(0, 5)
	return layout
	
if __name__ == "__main__":
	layout = np.zeros((15))
	#layout = np.array([0, 1, 2, 3, 1, 0, 0, 2, 1, 3, 0, 4, 0, 3, 0])
	#layout = generate_layout()
	circle = True

	# optimal strategy
	result = markovDecision(layout=layout, circle=circle)

	print("Snake and Ladder simulation with MDP")
	print("====================================")

	print(f"Generated Layout: {layout}")
	print(f"Expected cost for each cell: {result[0]}")
	print(f"Best dice for each cell: {result[1]}")

	# empirical simulation
	simulation = Simulation(
		layout=layout, 
		dice=DICE,
		circle=circle
	)
	empirical_costs = simulation.simulate(expec=result[0], strategy=StrategyType.OPTIMAL)
	print(f"Empirical cost for each cell: {empirical_costs}")
import numpy as np
import numpy.typing as npt

from src.MarkovDecisionProcess import MarkovDecisionProcess
from src.Simulation import Simulation

from utils.utils import generate_layout
from utils.common import DICE, StrategyType

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
	results = mdp.launch_iteration_value()
	return results
	
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
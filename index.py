import numpy as np
import numpy.typing as npt

from src.MarkovDecisionProcess import MarkovDecisionProcess
from src.Simulation import Simulation

from utils.plots import compare_costs
from utils.layouts import generate_layout, CUSTOM_LAYOUTS
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
	random_layout = generate_layout()
	custom_layout = CUSTOM_LAYOUTS["JAILS_ON_FAST_LANE"]
	circle = False

	# optimal strategy
	result = markovDecision(layout=custom_layout, circle=circle)
	expected_costs = result[0]
	best_dice = result[1]

	print("Snake and Ladder simulation with MDP")
	print("====================================")

	print(f"Generated Layout: {custom_layout}")
	print(f"Expected cost for each cell: {expected_costs}")
	print(f"Best die for each cell: {best_dice}")

	# empirical simulation
	simulation = Simulation(
		layout=custom_layout, 
		dice=DICE,
		circle=circle
	)
	empirical_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.OPTIMAL, 
		number_of_simulations=100000
		)
	print(f"Empirical cost for each cell: {empirical_costs}")

	compare_costs(
		layout=custom_layout,
		theoretical_costs=expected_costs,
		empirical_costs=empirical_costs,
		title=f"comparison of costs (circle={circle})"
	)
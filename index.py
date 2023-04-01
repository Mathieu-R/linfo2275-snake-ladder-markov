import numpy as np
import numpy.typing as npt
import click

from click.decorators import option
from click.types import Choice

from src.MarkovDecisionProcess import MarkovDecisionProcess
from src.Simulation import Simulation

from utils.plots import compare_costs_plot, compare_strategies_plot
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

	mdp.compute_adjacent_matrices()
	results = mdp.launch_iteration_value()
	return results

def compare_costs(layout_name: str, layout: npt.NDArray, best_dice, expected_costs, simulations: int, circle: bool):
	# empirical simulation
	simulation = Simulation(
		layout=layout, 
		dice=DICE,
		circle=circle
	)
	empirical_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.OPTIMAL, 
		number_of_simulations=simulations
		)
	print(f"Empirical cost for each cell: {empirical_costs}")

	compare_costs_plot(
		layout=layout,
		theoretical_costs=expected_costs,
		empirical_costs=empirical_costs,
		title=f"Comparison of costs (circle={circle})",
		subtitle=f"Layout: {layout_name}"
	)

def compare_strategies(layout_name: str, layout: npt.NDArray, best_dice, expected_costs, simulations: int, circle: bool):
	# empirical simulation
	simulation = Simulation(
		layout=layout, 
		dice=DICE,
		circle=circle
	)

	security_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.SECURITY, 
		number_of_simulations=simulations
		)
	
	normal_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.NORMAL, 
		number_of_simulations=simulations
		)
	
	risky_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.RISKY, 
		number_of_simulations=simulations
		)

	security_normal_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.SECURITY_NORMAL, 
		number_of_simulations=simulations
		)
	
	security_risky_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.SECURITY_RISKY, 
		number_of_simulations=simulations
		)
	
	normal_risky_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.NORMAL_RISKY, 
		number_of_simulations=simulations
		)
	
	random_costs = simulation.simulate(
		best_dice=best_dice, 
		strategy=StrategyType.RANDOM, 
		number_of_simulations=simulations
		)

	# check: https://matplotlib.org/stable/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py for a list of colors
	suboptimal_costs = {
		"security": {
			"data": security_costs,
			"color": "forestgreen",
			"label": "security die"
		},
		"normal": {
			"data": normal_costs,
			"color": "darkorange",
			"label": "normal die"
		},
		"risky": {
			"data": risky_costs,
			"color": "red",
			"label": "risky die"
		},
		"security_normal": {
			"data": security_normal_costs,
			"color": "yellowgreen",
			"label": "security & normal dice"
		},
		"security_risky": {
			"data": security_risky_costs,
			"color": "teal",
			"label": "security & risky dice"
		},
		"normal_risky": {
			"data": normal_risky_costs,
			"color": "chocolate",
			"label": "normal & risky dice"
		},
		"random": {
			"data": random_costs,
			"color": "silver",
			"label": "random die"
		}
	}
	
	compare_strategies_plot(
		layout=layout,
		optimal_costs=expected_costs,
		suboptimal_costs=suboptimal_costs,
		title=f"Comparison with suboptimal strategies (circle={circle})",
		subtitle=f"Layout: {layout_name}"
	)

@click.command()
@click.option(
	"--layout", "-l",
	type=click.Choice(["RANDOM", "NO_TRAPS", "JAILS_ON_FAST_LANE", "GAMBLE_EVERYWHERE", "NO_TRAPS_SLOW_LANE"]),
	default="NO_TRAPS",
	show_default=True,
	help="Type of game board"
)
@click.option(
	"--simulations", "-s",
	type=click.INT,
	default=10000,
	show_default=True,
	help="Number of simulations to run"
)
@click.option(
	"--circle", "-c",
	is_flag=True,
	help="Make the board circle"
)
@click.option(
	"--mdp_relevance_plot", "-mdp",
	is_flag=True,
	help="Show the relevance of MDP by empirical simulations."
)
@click.option(
	"--strategies_plot", "-sp",
	is_flag=True,
	help="Compare the optimal strategy with suboptimal ones."
)
def main(layout, simulations, circle, mdp_relevance_plot, strategies_plot):
	if layout == "RANDOM":
		custom_layout = generate_layout()
	else:
		custom_layout = CUSTOM_LAYOUTS[layout]
	
	# optimal strategy
	result = markovDecision(layout=custom_layout, circle=circle)
	expected_costs = result[0]
	best_dice = result[1]

	print("Snake and Ladder simulation with MDP")
	print("====================================")

	print(f"Generated Layout: {custom_layout}")
	print(f"Expected cost for each cell: {expected_costs}")
	print(f"Best die for each cell: {best_dice}")

	if mdp_relevance_plot:
		compare_costs(
			layout_name=layout,
			layout=custom_layout,
			best_dice=best_dice,
			expected_costs=expected_costs,
			simulations=simulations,
			circle=circle
		)
	
	elif strategies_plot:
		compare_strategies(
			layout_name=layout,
			layout=custom_layout,
			best_dice=best_dice,
			expected_costs=expected_costs,
			simulations=simulations,
			circle=circle
		)

if __name__ == "__main__":
	main()




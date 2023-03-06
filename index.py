from sys import last_traceback
import numpy as np
import numpy.typing as npt

from typing import List, Tuple
from src.BoardGame import BoardGame
from src.Die import Die, DieType
from utils.constants import GAMMA, INITIAL_DELTA, EPSILON


def markovDecision(layout: npt.NDArray, circle: bool = False) -> List[npt.NDArray]:
	"""launch the markov decision algorithm process to determine optimal strategy regarding 
	the choice of the dice in the snake and ladder games using the "value iteration" method.

	Args:
		layout (npt.NDArray): represents the layout of the game, each index represent a square and each value represent 
		circle (bool): indicate if the player must land exactly on the final square (circle = true) or still win 
			by overstepping the final square (circle = false)

	Returns:
		List[npt.NDArray, npt.NDArray]: a list containing two vectors as numpy arrays: Expec and Dice
	"""
	board = BoardGame(
		layout=layout, 
		dice=[
			Die(type=DieType.SECURITY, moves=[0, 1], trap_triggering_probability=0.0),
			Die(type=DieType.NORMAL, moves=[0, 1, 2], trap_triggering_probability=0.5),
			Die(type=DieType.RISKY, moves=[0, 1, 2, 3], trap_triggering_probability=1.0)
		], 
		circle=circle
	)

	# expected cost associated to the 14 squares of the game (excluding the goal square)
	Expec = np.array([0 for i in range(0, layout_size - 1)])
	# choice of the best dice for each of the 14 squares (excluding the goal square)
	Dice = np.array([1 for i in range(0, layout_size - 1)])
	


	# initialize every cell with 0 cost
	V0 = [0 for i in range(0, 15)]


	delta = INITIAL_DELTA
	while delta > EPSILON:
		Vk = V0.copy()
		delta = 0

		# start from the first cell
		state_idx = 0


		
		# walk through the game map [0,...,14]
		while state_idx < 15:
			manage_trap()

			costs = []

			for dice in DICES.keys():

				cost = 0
				for move in DICES[dice]["moves"]:
					# if circle map and player overstep the final cell, restart from start
					if circle and state_idx + move > 14:
						next_state_idx = 0
					
					elif state_idx + move >= 14:
						won = True

					cost += (1 / len(DICES[dice]["moves"])) * (??? + GAMMA * Vk[state_idx + move])

			Vk[state_idx] = max(costs)
	
	return [np.array([]), np.array([])]

def update_bellman_function(initial_cell, dice_type):
	V = 1

	dice = dices[dice_type]
	for move in dice["moves"]:
		make_move(initial_cell=initial_cell, move=move)



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
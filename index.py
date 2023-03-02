from typing import List

import numpy as np
import numpy.typing as npt

GAMMA = 0.9
EPSILON = 10e-3

def markovDecision(layout: npt.NDArray, circle: bool) -> List[npt.NDArray]:
	"""launch the markov decision algorithm process to determine optimal strategy regarding 
	the choice of the dice in the snake and ladder games using the "value iteration" method.

	Args:
		layout (npt.NDArray): represents the layout of the game, each index represent a square and each value represent 
		circle (bool): indicate if the player must land exactly on the final square (circle = true) or still win 
			by overstepping the final square (circle = false)

	Returns:
		List[npt.NDArray]: a list containing two vectors: Expec and Dice
	"""
	dices = {
		"security": {
			"moves": [0, 1],
			"trap_trig_prob": 0
		},
		"normal": {
			"moves": [0, 1, 2],
			"trap_trig_prob": 0.5
		},
		"risky": {
			"moves": [0, 1, 2, 3],
			"trap_trig_prob": 1
		}
	}

	won = False

	# initialize every cell with 0 cost
	V0 = [0 for i in range(0, 15)]

	while not won:
		Vk = V0.copy()
		delta = 0

		# start from the first cell
		state_idx = 0


		
		# walk through the game map [0,...,14]
		while state_idx < 15:
			# check the reward (i.e. trap)
			reward_type = layout[state_idx]

			if reward_type == 1:
				# teleport back to 1st square (restart)
				state_idx = 1
			elif reward_type == 2:
				# teleport 3 steps backward (penalty)
				state_idx = max(0, state_idx - 3)
			elif reward_type == 3:
				# wait one turn before playing again (prison)
				continue
			elif reward_type == 4:
				# randomly teleport anywhere on the board (gamble)
				state_idx = np.random.randint(0, 15)

			costs = []

			for dice in dices.keys():

				cost = 0
				for move in dices[dice]["moves"]:
					# if circle map and player overstep the final cell, restart from start
					if circle and state_idx + move > 14:
						next_state_idx = 0
					
					elif state_idx + move >= 14:
						won = True

					cost += (1 / len(dices[dice]["moves"])) * (??? + GAMMA * Vk[state_idx + move])

			Vk[state_idx] = max(costs)
	
	return [np.array([]), np.array([])]


if __name__ == "__main__":
	layout: npt.NDArray = np.array([0, 1, 2, 3, 1, 0, 0, 2, 1, 3, 0, 4, 0, 3, 0])
	markovDecision(layout=layout, circle=True)
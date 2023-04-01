import numpy as np 

from .common import TrapType

def generate_layout():
	layout = np.zeros((15), dtype=int)
	for i in range(1, 14):
		# generate a cell of type between 0 and 4
		# first and last cell are excluded
		layout[i] = np.random.randint(0, 5)
	#layout = np.array([0, 1, 1, 4, 2, 1, 0, 1, 0, 1, 0, 1, 0, 4, 0])
	return layout

def layout_jails_on_fast_lane():
	layout = generate_layout()
	layout[10:14] = TrapType.PRISON.value
	#return np.array([0, 1, 4, 0, 3, 0, 2, 0, 0, 0, 3, 3, 3, 3, 0])
	return layout


def layout_gamble_everywhere():
	layout = np.zeros((15), dtype=int)
	layout[1:14] = TrapType.GAMBLE.value
	return layout

def layout_traps_everywhere_except_slow_lane():
	layout = np.zeros((15), dtype=int)
	layout[1] = TrapType.PRISON.value
	layout[2] = TrapType.RESTART.value
	layout[3:10] = TrapType.NONE.value
	layout[10] = TrapType.PENALTY.value
	layout[11] = TrapType.GAMBLE.value
	layout[12] = TrapType.RESTART.value
	layout[12] = TrapType.PRISON.value
	return layout

CUSTOM_LAYOUTS = {
	"NO_TRAPS": np.zeros((15)),
	"JAILS_ON_FAST_LANE": layout_jails_on_fast_lane(),
	"GAMBLE_EVERYWHERE": layout_gamble_everywhere(),
	"NO_TRAPS_SLOW_LANE": layout_traps_everywhere_except_slow_lane()
}
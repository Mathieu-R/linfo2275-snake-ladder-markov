import numpy as np 

from .common import TrapType

def generate_layout():
	layout = np.zeros((15))
	for i in range(1, 14):
		# generate a cell of type between 0 and 4
		# first and last cell are excluded
		layout[i] = np.random.randint(0, 5)
	return layout

def layout_jails_on_fast_lane():
	layout = generate_layout()
	layout[10:13] = TrapType.PRISON.value
	return layout

def layout_gamble_everywhere():
	layout = np.zeros((15))
	layout[1:13] = TrapType.GAMBLE.value
	return layout

def layout_traps_everywhere_except_slow_lane():
	layout = np.zeros((15))
	layout[1] = TrapType.PRISON.value
	layout[2] = TrapType.RESTART.value
	layout[3:9] = TrapType.NONE.value
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
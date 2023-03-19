import numpy as np
import numpy.typing as npt 

def find_indices(array: npt.NDArray, item: int) -> npt.NDArray:
	return np.where(array == item)[0]

def generate_layout():
	layout = np.zeros((15))
	for i in range(1, 14):
		# generate a cell of type between 0 and 4
		# first and last cell are excluded
		layout[i] = np.random.randint(0, 5)
	return layout
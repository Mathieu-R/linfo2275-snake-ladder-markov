import numpy as np
import numpy.typing as npt 

def find_indices(array: npt.NDArray, item: int) -> npt.NDArray:
	return np.where(array == item)[0]
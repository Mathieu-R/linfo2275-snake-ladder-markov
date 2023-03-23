import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from matplotlib import ticker

def compare_costs(layout: npt.NDArray, theoretical_costs: npt.NDArray, empirical_costs: npt.NDArray, title: str):
	fig, ax = plt.subplots(layout="constrained")

	n = len(layout)
	x = np.arange(1, n)
	width = 0.3

	ax.bar(x, theoretical_costs, width=width, color="royalblue", align="edge", label="MDP cost")
	ax.bar(x + width, empirical_costs, width=width, color="orange", align="edge", label="empirical cost")

	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

	ax.set_xlabel("State")
	ax.set_ylabel("Expected cost")
	ax.set_title(title)

	ax.legend(loc="upper right", ncols=3)

	plt.show()
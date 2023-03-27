import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from matplotlib import ticker

def compare_costs_plot(layout: npt.NDArray, theoretical_costs: npt.NDArray, empirical_costs: npt.NDArray, title: str, subtitle: str):
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

def compare_strategies_plot(layout: npt.NDArray, optimal_costs: npt.NDArray, suboptimal_costs: dict, title: str, subtitle: str):
	fig, ax = plt.subplots(layout="constrained")

	n = len(layout)
	x = np.arange(1, n)

	ax.plot(x, optimal_costs, ls="--", color="royalblue", label="optimal", zorder=-1)

	for costs in suboptimal_costs.values():
		ax.plot(x, costs["data"], color=costs["color"], label=costs["label"])

	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

	ax.set_xlabel("State")
	ax.set_ylabel("Expected cost")
	ax.set_title(title)

	ax.legend(loc="upper right", ncols=3)

	plt.show()

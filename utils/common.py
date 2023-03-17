from enum import Enum
from src.Die import Die, DieType

DICE = [
	Die(type=DieType.SECURITY, moves=[0, 1], trap_triggering_probability=0.0),
	Die(type=DieType.NORMAL, moves=[0, 1, 2], trap_triggering_probability=0.5),
	Die(type=DieType.RISKY, moves=[0, 1, 2, 3], trap_triggering_probability=1.0)
]

class TrapType(Enum):
	NONE = 0
	RESTART = 1
	PENALTY = 2
	PRISON = 3
	GAMBLE = 4

class CellType(Enum):
	STARTING_CELL = 0
	SLOW_LANE_FIRST_CELL = 3
	SLOW_LANE_LAST_CELL = 9
	FAST_LANE_FIRST_CELL = 10
	FAST_LANE_LAST_CELL = 13

class StrategyType(Enum):
	OPTIMAL = "optimal"
	SECURITY = "security"
	NORMAL = "normal"
	RISKY = "risky"
	RANDOM = "random"
	SECURITY_NORMAL = "security_and_normal"
	SECURITY_RISKY = "security_and_risky"
	NORMAL_RISKY = "normal_and_risky"
	SECURITY_OPTIMAL = "security_and_optimal"
	NORMAL_OPTIMAL = "normal_and_optimal"
	RISKY_OPTIMAL = "risky_and_optimal"
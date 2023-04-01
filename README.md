# snake-ladder-markov
Markov decision processes in the framework of snake and ladder game

### Installing
Clone the project
```bash
$ git clone https://github.com/Mathieu-R/celestial-mechanics
```

Create virtual environment
```bash
$ python3 -m venv <env-name>
$ source env/bin/activate # OSX / Unix
$ env\\Scripts\\activate.bat # Windows
$ python3 -m pip install --upgrade pip
```

Install required packages
```bash
$ python3 -m pip install click numpy matplotlib
```

### Launch simulations    
Type `--help` command to show all the possible parameters that are available and how to launch a simulation.

```bash
$ python3 index.py --help
Usage: index.py [OPTIONS]

Options:
  -l, --layout [RANDOM|NO_TRAPS|JAILS_ON_FAST_LANE|GAMBLE_EVERYWHERE|NO_TRAPS_SLOW_LANE]
                                  Type of game board  [default: NO_TRAPS]
  -s, --simulations INTEGER       Number of simulations to run  [default:
                                  10000]
  -c, --circle                    Make the board circle
  -mdp, --mdp_relevance_plot      Show the relevance of MDP by empirical
                                  simulations.
  -sp, --strategies_plot          Compare the optimal strategy with suboptimal
                                  ones.
  --help                          Show this message and exit.
```

For example: 
```bash
$ python3 index.py -l NO_TRAPS -sp -c
```
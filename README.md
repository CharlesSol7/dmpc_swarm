# DMPC Swarm

This package allows to find path for multi-agents swarms using DMPC.
This version is executed offline and generate the path in a csv file. The simulation can be run with the plot_csv.py file

The path_planning alogrithm is implemented in C++, using eigen library and eigen-quadprog solver. The algorithm is also implemented in Python, using OPTCVX solver.

## Getting Started

### Prerequisites

For C++, those libraries need to be installed:

```
ros2
yaml-cpp
Eigen
eigen-quadprog
```

For Python, those packages need to be installed:

```
yaml
numpy
cvxopt
matplotlib
imageio
tgqm
pathlib
```

### Installing

Install all dependencies
Copy the dmpc_swarm folder in your ros2_ws/src directory
If you are installing dmpc_swarm in another directory, you will have to change the path in path_planning.cpp
Then build the package in ros2_ws running the command:

```
cd ros2_ws
colcon build --symlink-install 
```


## Running the tests

Set the desired parameters of the simulation in the following file.
```
dmpc_swarm/dmpc_swarm/config_swarm.yaml
```

For the Python code, run the python script path_planning.py. A numpy file containing the optimal trajectories of alla agents is generated. The 3D animation of the simulation is automatically displayed.

For the C++ code, run the following command:
```
ros2 run dmpc_swarm path_planning
cd ros2_ws/src/dmpc_swarm/dmpc_swarm/
python3 plot/csv.py
```
The C++ path planning code generate a csv file containing the trajectories of all agents.

The track_offline_npy (for npy path file) and track_offline_csv (for csv path file) file allow to control a swarm of Crazyflies from the Crazyswarm2 package.

## Authors

* **Charles Sol** - *Initial work* - [CharlesSol7](https://github.com/CharlesSol7/dmpc_swarm)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details



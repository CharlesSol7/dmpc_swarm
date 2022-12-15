DMPC_swarm package

Author: Charles Sol
email: charles.sol@mail.utoronto.ca

This package allows to find path for multi-agents swarms using DMPC.
This version is executed offline and generate the path in a csv file. The simulation can be run with the plot_csv.py file

The path_planning alogrithm is implemented in C++, using eigen library and eigen-quadprog solver. The algorithm is also implemented in Python, using OPTCVX solver.

Installation

You musrt install dependency package (Eigen, eigen_quadprog, Yaml)
First copy the dmpc_swarm folder in the ros2_ws/src directory
Then build the package in ros2_ws running the command:
	colcon build --symlink-install 

Configuration file with parameters: dmpc_swarm/dmpc_swarm/config_swarm.yaml
Run DMPC algorithm from Home directory with command:
	ros2 run dmpc_swarm path_planning

Then visualize simulation running the file plot_csv.py from dmpc_swarm/dmpc_swarm/ directory:
A gif is generated

the track_offline_csv allows to read the csv path file to control a swarm of Crazyflies from the Crazyswarm2 package.

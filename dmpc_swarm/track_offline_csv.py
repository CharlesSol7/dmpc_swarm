#!/usr/bin/env python

import numpy as np
from pathlib import Path
import yaml

from crazyflie_py import *

def normalize(v):
    norm = np.linalg.norm(v)
    assert norm > 0
    return v / norm

def executeTrajectory(timeHelper, allcfs, path, dt=0.01, offset=np.zeros(3)):
    # array path [n_agent, n_derivative, xyz, timestep]
    yaw = 0.0
    dyaw = 0.0
    start_time = timeHelper.time()
    n_cf = len(allcfs.crazyflies)
    while not timeHelper.isShutdown():
        t = timeHelper.time() - start_time
        k = int(t/dt)
        if k >= path.shape[3]:
            break
        
        for i, cf in enumerate(allcfs.crazyflies):
            timeHelper.sleepForRate(n_cf/dt)
            jerk = path[i,3,:,k]

            thrust = path[i,2,:,k] + np.array([0, 0, 9.81]) # add gravity

            z_body = normalize(thrust)
            x_world = np.array([np.cos(yaw), np.sin(yaw), 0])
            y_body = normalize(np.cross(z_body, x_world))
            x_body = np.cross(y_body, z_body)

            jerk_orth_zbody = jerk - (np.dot(jerk, z_body) * z_body)
            h_w = jerk_orth_zbody / np.linalg.norm(thrust)

            omega = np.array([-np.dot(h_w, y_body), np.dot(h_w, x_body), z_body[2] * dyaw])
            
            cf.cmdFullState(
                path[i,0,:,k] + offset, #pos
                path[i,1,:,k],          #vel
                path[i,2,:,k],          #acc
                yaw,                    #yaw
                omega)                  #omega


def main():
    with open(Path(__file__).parent /'config_swarm.yaml') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    num_drone = param["num_drone"]
    n_derivative = param["n_derivative"]
    dt_upd = param["dt_upd"]

    
    path_csv = np.genfromtxt(Path(__file__).parent / 'path.csv', delimiter=', ')
    #num_drone = int(path_csv.shape[0]/(3*(n_derivative+1)))
    path = np.zeros((num_drone, n_derivative+1, 3,path_csv.shape[1]))
    r=0
    for i in range(0,num_drone):
        for d in range(n_derivative+1):
            path[i,d] = path_csv[3*r:3*(r+1), :]
            r+=1
    
    Z = 1.0
    print("Crazyswarm initialization")
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    allcfs = swarm.allcfs

    timeHelper.sleep(0.2)
    for i, cf in enumerate(allcfs.crazyflies):
        cf.takeoff(targetHeight=Z, duration=1.5+Z)
    print('takeoff')
    timeHelper.sleep(2+Z)

    for i, cf in enumerate(allcfs.crazyflies):
        pos = path[i,0,:,0]
        print(i, pos)
        cf.goTo(pos, 0, 2.0)

    timeHelper.sleep(2+Z)

    executeTrajectory(timeHelper, allcfs, path, dt = dt_upd)
    
    print('landing')
    for sendn in range(10):
        allcfs.land(targetHeight=0.02, duration=4.0+Z)
    timeHelper.sleep(5.0+Z)


if __name__ == "__main__":
    main()

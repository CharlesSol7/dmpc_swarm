#from ossaudiodev import control_labels
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits import mplot3d
import imageio.v2 as imageio
import yaml
import os
from pathlib import Path

def plotRealTime(Path_full, Points, Obstacles, quad, dt, x_lim, y_lim, z_lim, Path_horizon_all = None, Dt = None, speed=1, loop=False):    
    plt.ion()
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 100)
    color = ['r','b','y','g','c','m','k']
    t0 = time.time_ns()/10**9
    t=0
    k_mpc = 0
    write_gif = True
    writer = imageio.get_writer(Path(__file__).parent / ("simulation" + str(Path_full.shape[0]) + "agents.gif"), mode='I')
    angle = 0
    while t < Path_full.shape[2]*dt:
        t = (time.time_ns()/10**9-t0)*speed
        ax.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(x_lim[0],x_lim[1])
        ax.set_ylim(y_lim[0],y_lim[1])
        ax.set_zlim(0,z_lim[1])
        for obstacle in Obstacles:
            x = obstacle[0,0] + obstacle[1,0] * np.outer(np.cos(u), np.sin(v))
            y = obstacle[0,1] + obstacle[1,1] * np.outer(np.sin(u), np.sin(v))
            z = obstacle[0,2] + obstacle[1,2] * np.outer(np.ones_like(u), np.cos(v))
            z[z<0]= np.nan
            z[z>z_lim[1]]= np.nan
            x[z<0]= np.nan
            x[z>z_lim[1]]= np.nan
            y[z<0]= np.nan
            y[z>z_lim[1]]= np.nan
            ax.plot_surface(x, y, z, color='y', alpha = 0.4)
        ax.plot3D(Points[:,0], Points[:,1], Points[:,2], 'go')
        k = min(int(t/dt), Path_full.shape[2]-1)
        
        for i in range(0,Path_full.shape[0]):
            ax.plot3D(Path_full[i,0,0:k], Path_full[i,1,0:k], Path_full[i,2,0:k], color[i%7])
            x = Path_full[i,0,k] + quad[0,0] * np.outer(np.cos(u), np.sin(v))
            y = Path_full[i,1,k] + quad[1,1] * np.outer(np.sin(u), np.sin(v))
            z = Path_full[i,2,k] + quad[2,2] * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, color=color[i%7], alpha = 0.4)

            if Path_horizon_all is not None:
                k_mpc = min(int(t/Dt),Path_horizon_all.shape[0]-1)
                ax.plot3D(Path_horizon_all[k_mpc,i,0,:], Path_horizon_all[k_mpc,i,1,:], Path_horizon_all[k_mpc,i,2,:],  color[i%7]+'--')

        ax.view_init(elev=30*np.sin(angle*np.pi/180)+40, azim=angle)
        angle +=1
        plt.draw()
        plt.pause(0.0001)
        if plt.fignum_exists(1) == False:
            break
        if write_gif:
            filename = 'path_'+str(k)+'.png'
            plt.savefig(filename)
            image = imageio.imread(filename)
            writer.append_data(image)
            os.remove(filename)
        if loop and(t > Path_full.shape[2]*dt):
            write_gif = False
            plt.pause(1)
            t0 = time.time_ns()/10**9
            t=0
            k_mpc = 0

    plt.pause(1)


if __name__ == "__main__":
    with open(Path(__file__).parent /'config_swarm.yaml') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    num_drone = param["num_drone"]
    num_obs = param["num_obs"]
    config = param["config"]

    n_order = param["n_order"]
    n_derivative = param["n_derivative"]
    dt_upd = param["dt_upd"]
    dt_opt = param["dt_opt"]
    dt_mpc = param["dt_mpc"]
    T_hor = param["T_hor"]
    T_max = param["T_max"]

    x_lim  = param["x_lim"]
    y_lim  = param["y_lim"]
    z_lim  = param["z_lim"]
    #z_lim  = [0, 1.5]

    env_drone  = param["env_drone"]
    r_min = (env_drone[0]**2+env_drone[1]**2+env_drone[2]**2)**0.5
    theta = np.array([[env_drone[0]/r_min,0,0], [0,env_drone[1]/r_min,0], [0,0,env_drone[2]/r_min]])

    init_drone = param["init_drone"]
    goal_drone = param["goal_drone"]
    pd = np.zeros((num_drone,3))
    if config == 'circle':
        Radius = param["radius"]
        altitude = param["height"]
        for i in range(num_drone):
            thetai = i* 2*np.pi/num_drone
            if num_drone==1 :
                thetaf = np.pi
            else:
                thetaf = (i+int(num_drone/2))* 2*np.pi/num_drone
            pd[i] = np.array([Radius * np.cos(thetaf),Radius * np.sin(thetaf), altitude])
    elif config == 'custom':
        for i in range(num_drone):
            pd[i] = [goal_drone[i][0], goal_drone[i][1], goal_drone[i][2]]

    pos_static_obs = param["pos_static_obs"]
    dim_static_obs = param["dim_static_obs"]
    obstacles = np.zeros((num_obs,2,3))
    for i in range(num_obs):
        obstacles[i,0] = pos_static_obs[i]
        obstacles[i,1] = dim_static_obs[i]

    path_csv = np.genfromtxt(Path(__file__).parent / 'path.csv', delimiter=', ')
    num_drone = int(path_csv.shape[0]/(3*(n_derivative+1)))
    path = np.zeros((num_drone, n_derivative+1, 3, path_csv.shape[1]))
    r=0
    for i in range(0,num_drone):
        for d in range(n_derivative+1):
            path[i,d] = path_csv[3*r:3*(r+1), :]
            r+=1

    N_mpc = int(T_max/dt_mpc)-1
    path_hor_csv = np.genfromtxt(Path(__file__).parent / 'path_horizon.csv', delimiter=', ')
    path_horizon = np.zeros((N_mpc, num_drone, 3,path_hor_csv.shape[1]))
    r = 0
    for n in range(0, N_mpc):
        for i in range(0,int(num_drone)):
            path_horizon[n,i] = path_hor_csv[3*r:3*(r+1),:]
            r+=1

    #obstacles = np.array([])
    plotRealTime(path[:,0,:,:], pd, obstacles, theta*r_min, dt_upd, x_lim, y_lim, z_lim, Path_horizon_all = path_horizon, Dt=dt_mpc, speed = 0.1, loop=True)

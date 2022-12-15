# Author: Charles Sol

import numpy as np
import cvxopt
import matplotlib.pyplot as plt
import time
from mpl_toolkits import mplot3d
import imageio.v2 as imageio
from tqdm import tqdm
import os
from pathlib import Path
import yaml



#----------------------Bezier curves-----------------------------
def computeBinominal(n, k):
    value = 1.0
    for i in range(1,k+1):
        value *= ((n + 1 - i) / i)
    if (n == k):
        value = 1
    return int(value)

def computeBezierCoefficient(t, T, n_order, n_derivative):
    bezierCoef = np.zeros((n_derivative+1,n_order+1))
    for i in range(0,n_order+1):
        for d in range(0,n_derivative+1):
            if i <= n_order-d:
                Coef = computeBinominal(n_order-d, i)* (1 - t/T)**(n_order -d- i) * (t/T)**i * (1/T)**d
                for k in range(n_order-d+1, n_order+1):
                    Coef *= k
                for j in range(0,d+1):
                    bezierCoef[d,j+i] += Coef*computeBinominal(d, j)*(-1)**(j+d)
    return bezierCoef # Columns are coefficient of each control point and rows for each derivative

def computeBezierCurve(Points, T, dt_opt, n_derivative=0):
    T = int(T/dt_opt)*dt_opt
    Curve = np.zeros((n_derivative+1,3,int(T/dt_opt)+1))
    for k,t in enumerate(np.arange(0,T+dt_opt, dt_opt)):
        bezierCoef = computeBezierCoefficient(t, T, Points.shape[1]-1,n_derivative)
        for d in range(0,n_derivative+1):
            Curve[d][:,[k]] = Points @ bezierCoef[[d],:].T

    return Curve

def plotRealTime(Path_full, Points, Obstacles, quad, dt_upd, Path_horizon_all = None, dt_mpc = None, speed=1, loop=False):    
    plt.ion()
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    color = ['r','b','y','g','c','m','k']
    t0 = time.time_ns()/10**9
    t=0
    k_mpc = 0
    write_gif = True
    writer = imageio.get_writer('simulation.gif', mode='I')
    angle = 0
    while t < Path_full.shape[2]*dt_upd:
        t = (time.time_ns()/10**9-t0)*speed
        ax.cla()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-2.5,2.5)
        ax.set_ylim(-2.5,2.5)
        ax.set_zlim(0,3)
        for obstacle in Obstacles:
            x = obstacle[0,0] + obstacle[1,0] * np.outer(np.cos(u), np.sin(v))
            y = obstacle[0,1] + obstacle[1,1] * np.outer(np.sin(u), np.sin(v))
            z = obstacle[0,2] + obstacle[1,2] * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, color='y', alpha = 0.4)
        ax.plot3D(Points[:,0], Points[:,1], Points[:,2], 'go')
        k = min(int(t/dt_upd), Path_full.shape[2]-1)
        
        for i in range(0,Path_full.shape[0]):
            ax.plot3D(Path_full[i,0,0:k], Path_full[i,1,0:k], Path_full[i,2,0:k], color[i%7])
            x = Path_full[i,0,k] + quad[0,0] * np.outer(np.cos(u), np.sin(v))
            y = Path_full[i,1,k] + quad[1,1] * np.outer(np.sin(u), np.sin(v))
            z = Path_full[i,2,k] + quad[2,2] * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(x, y, z, color=color[i%7], alpha = 0.4)

            if Path_horizon_all is not None:
                k_mpc = min(int(t/dt_mpc),Path_horizon_all.shape[0]-1)
                ax.plot3D(Path_horizon_all[k_mpc,i,0,:], Path_horizon_all[k_mpc,i,1,:], Path_horizon_all[k_mpc,i,2,:],  color[i%7]+'--')

        ax.view_init(elev=45*np.sin(angle*np.pi/180)+20, azim=angle)
        angle+=1
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
        if loop and(t > Path_full.shape[2]*dt_upd):
            write_gif = False
            plt.pause(1)
            t0 = time.time_ns()/10**9
            t=0
            k_mpc = 0

    plt.pause(1)

#----------------------DMPC-----------------------------
def initDMPC_pos(num_agent, Obstacles, param, constraints, k_delay=0):
    n_order, n_derivative, dt_opt, T_hor, L, theta, r_min, safe_dist, kappa, weight_goal, weight_quad, weight_line = param
    constraints_on_derivative, alpha_energy = constraints
    
    T_hor = int(T_hor/dt_opt)*dt_opt
    num_obst = Obstacles.shape[0]
    n_epsilon = num_agent*num_obst + num_agent*(num_agent-1)
    n_var = 3*num_agent*(L*(n_order)+1) + n_epsilon
    K_bezier = int(T_hor/dt_opt)
    K = L*K_bezier

    C = np.zeros((n_derivative+1, 3*K*num_agent, n_var))
    Q = np.zeros((3*num_agent*(K),3*num_agent*(K)))
    
    Aeq = np.zeros((3*(num_agent*(n_derivative)*(L-1)),n_var))
    beq = np.zeros((3*(num_agent*(n_derivative)*(L-1)),))
    
    Ain = np.zeros((3*K*num_agent*(2*len(constraints_on_derivative[0])),n_var))
    bin = np.zeros((3*K*num_agent*(2*len(constraints_on_derivative[0])),))

    coef_end_point = computeBezierCoefficient(T_hor, T_hor, n_order, n_derivative)
    
    theta_quad_inv = np.linalg.inv(theta)
    theta_quad_inv2 = np.linalg.matrix_power(theta_quad_inv,2)
    Theta_quad_inv2 = np.zeros((3*(K-k_delay),3*(K-k_delay)))
    theta_obst_inv = np.zeros((num_obst, 3,3))
    theta_obst_inv2 = np.zeros((num_obst, 3,3))
    Theta_obst_inv2 = np.zeros((num_obst, 3*K,3*K))
    for no, obstacle in enumerate(Obstacles):
        theta_obst_inv[no] = np.linalg.inv(np.array([[obstacle[1,0]/r_min+theta[0,0],0,0], [0,obstacle[1,1]/r_min+theta[1,1],0],[0,0,obstacle[1,2]/r_min+theta[2,2]]])/2)
        theta_obst_inv2[no] = np.linalg.matrix_power(np.linalg.inv(np.array([[obstacle[1,0]/r_min+theta[0,0],0,0], [0,obstacle[1,1]/r_min+theta[1,1],0],[0,0,obstacle[1,2]/r_min+theta[2,2]]])/2),2)
        
    for k in range(0,K):
        if k < K-k_delay:
            Theta_quad_inv2[3*k:3*(k+1),3*k:3*(k+1)] = theta_quad_inv2
        for ob in range(0,num_obst):
            Theta_obst_inv2[ob][3*k:3*(k+1),3*k:3*(k+1)] = theta_obst_inv2[ob]
        if int(k/K_bezier) < L:
            l = int(k/K_bezier)
        t = (k%K_bezier)*dt_opt
        coef = computeBezierCoefficient(t, T_hor, n_order, n_derivative)
        for i in range(0,num_agent):
            if k>K-1-kappa:
                Q[3*(K*i+k):3*(K*i+k+1),3*(K*i+k):3*(K*i+k+1)] = np.eye(3)*weight_goal
            for n in range(0,n_order+1):
                for m, d in enumerate(constraints_on_derivative[0]): # Physical constraints
                    Ain[3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m):3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+1),3*(i*(L*n_order+1)+ (l*n_order)+n):3*(i*(L*n_order+1)+ (l*n_order)+n+1)] = -coef[[d],n]*np.eye(3)
                    bin[3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m):3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+1)] = -constraints_on_derivative[1][m][0:3]
                    Ain[3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+1):3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+2),3*(i*(L*n_order+1)+ (l*n_order)+n):3*(i*(L*n_order+1)+ (l*n_order)+n+1)] = coef[[d],n]*np.eye(3)
                    bin[3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+1):3*(i*(2*(K*len(constraints_on_derivative[0]))) + k*2*len(constraints_on_derivative[0])+2*m+2)] = constraints_on_derivative[1][m][3:6]
                for d in range(0,n_derivative+1):
                    C[d][3*(k+K*i):3*(k+K*i+1), 3*(i*(L*n_order+1)+l*n_order+n):3*(i*(L*n_order+1)+l*n_order+n+1)] = coef[[d],n]*np.eye(3)
                    if (not k%K_bezier) and (l > 0) and (d>0): # Continuity constraint
                        Aeq[3*(i*(n_derivative)*(L-1)+(n_derivative)*(l-1)+(d-1)):3*(i*(n_derivative)*(L-1)+(n_derivative)*(l-1)+d),3*(i*(L*n_order+1)+(l-1)*n_order+n):3*(i*(L*n_order+1)+(l-1)*n_order+n+1)] += coef_end_point[[d],n]*np.eye(3)
                        Aeq[3*(i*(n_derivative)*(L-1)+(n_derivative)*(l-1)+(d-1)):3*(i*(n_derivative)*(L-1)+(n_derivative)*(l-1)+d),3*(i*(L*n_order+1)+l*n_order+n):3*(i*(L*n_order+1)+l*n_order+n+1)] -= coef[[d],n]*np.eye(3)
    
    J_quad = np.zeros((n_var, n_var))
    J_line = np.zeros((n_var,))
    for n in range(0,n_epsilon):
        J_quad[n_var-n_epsilon+n,n_var-n_epsilon+n] = weight_quad
        J_line[n_var-n_epsilon+n] = -weight_line

    J_quad += C[0].T @ Q @ C[0]
    for d in range(1,n_derivative):
        J_quad += alpha_energy[d-1]* (C[d].T @ C[d])
    QC = -np.dot(Q, C[0])

    qp_param = [C, Q, Ain, bin, Aeq, beq, J_quad, J_line, QC, theta_quad_inv, theta_quad_inv2, Theta_quad_inv2, theta_obst_inv, theta_obst_inv2, Theta_obst_inv2, k_delay]
    return qp_param

def computeDMPC_pos(X0, pd, Obstacles, param, qp_param, constraints, P_prev):
    n_order, n_derivative, dt_opt, T_hor, L, theta, r_min, safe_dist, kappa, q, weight_quad, weight_line = param
    [C, Q, Ain_der, bin_der, Aeq_cont, beq_cont, J_quad, J_line, QC, theta_quad_inv, theta_quad_inv2, Theta_quad_inv2, theta_obst_inv, theta_obst_inv2, Theta_obst_inv2, k_delay] = qp_param.copy()
    constraints_on_derivative, alpha_energy = constraints
    
    T_hor = int(T_hor/dt_opt)*dt_opt
    num_agent = X0.shape[0]
    num_obst = Obstacles.shape[0]
    n_epsilon = num_agent*num_obst + num_agent*(num_agent-1)
    n_var = 3*num_agent*(L*(n_order)+1) + n_epsilon
    n_ini_state_cstr = X0.shape[1]
    K_bezier = int(T_hor/dt_opt)
    K = L*K_bezier
    
    if P_prev is None:
        P_prev = np.zeros((n_var,1))
        for i in range(0,num_agent):
            P_prev[3*i*(L*n_order+1):3*(i+1)*(L*n_order+1), :] = np.tile(X0[i][0], (L*n_order+1)).reshape(-1,1)

    Aeq_ini = np.zeros((3*(num_agent*n_ini_state_cstr),n_var))
    beq_ini = np.zeros((3*(num_agent*n_ini_state_cstr),))
    
    Pd = np.zeros((3*K*num_agent,1))
    for i in range(0,num_agent):
        Pd[3*K*i:3*K*(i+1), :] = np.tile(pd[i].T, K).T.reshape(-1,1)

    xi = np.ones((K,num_agent*(num_agent-1) + num_agent*num_obst))*safe_dist
    n_col_cnstr = np.zeros((num_agent*(num_agent-1)+ num_agent*num_obst),dtype=int)
    
    coef = computeBezierCoefficient(0, T_hor, n_order, n_derivative)
    for i in range(0,num_agent):
        for n in range(0,n_order+1):
            for d in range(0,n_ini_state_cstr): # initial state constraints
                Aeq_ini[3*(i*n_ini_state_cstr+d):3*(i*n_ini_state_cstr+d+1),3*(i*(L*n_order+1)+n):3*(i*(L*n_order+1)+n+1)] = coef[[d],n]*np.eye(3)
                beq_ini[3*(i*n_ini_state_cstr+d):3*(i*n_ini_state_cstr+d+1)] = X0[i][d].flatten()
    
    for k in range(0,K):
        r=0
        if k < K-k_delay:
            for i in range(0,num_agent):
                for j in range(0,num_agent):
                    if i is not j:
                        xi[k,r]= np.linalg.norm(theta_quad_inv @ (C[0][3*(K*i+k):3*(K*i+k+1)]-C[0][3*(K*j+k+k_delay):3*(K*j+k+k_delay+1)]) @ P_prev)
                        if xi[k,r] < safe_dist:
                            n_col_cnstr[r] = n_col_cnstr[r]+1
                        r +=1
        for i in range(0,num_agent):
            for ob in range(0,num_obst):
                xi[k,r]= np.linalg.norm(theta_obst_inv[ob] @ (C[0][3*(K*i+k):3*(K*i+k+1)] @ P_prev -Obstacles[ob][0].reshape(-1,1)))
                if xi[k,r] < safe_dist:
                    n_col_cnstr[r] = n_col_cnstr[r]+1
                r +=1
                
    n=0
    Ain_col = np.zeros((np.sum(n_col_cnstr)+n_epsilon, n_var))
    bin_col = np.zeros((np.sum(n_col_cnstr)+n_epsilon,))

    for i in range(0,num_agent):
        for j in range(0,num_agent):
            if (i is not j):
                if n_col_cnstr[n]>0:
                    nu = Theta_quad_inv2 @ ((C[0][3*(K*i):3*(K*(i+1)-k_delay)]-C[0][3*(K*j+k_delay):3*(K*(j+1))]) @ P_prev)
                    
                    Nu = np.zeros((n_col_cnstr[n],3*(K-k_delay)))
                    q =0
                    xi_ij = np.zeros(n_col_cnstr[n],)
                    for k in range(0,K-k_delay):
                        if xi[k,n] < safe_dist:
                            Nu[q,3*k:3*(k+1)] = nu[3*k:3*(k+1)].flatten()
                            xi_ij[q]=xi[k,n]
                            q+=1
                    Ain_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1]),:] = -Nu @ C[0][3*(K*i):3*(K*(i+1)-k_delay)]
                    Ain_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1]),n_var-n_epsilon+n] = xi_ij
                    bin_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1])] = -r_min*xi_ij - xi_ij**2 - (Nu @ C[0][3*(K*i):3*(K*(i+1)-k_delay)] @ P_prev).flatten()

                    Ain_col[np.sum(n_col_cnstr)+n,n_var-n_epsilon+n] = 1
                    bin_col[np.sum(n_col_cnstr)+n] = 0
                n +=1
    for i in range(0,num_agent):
        for ob in range(0,num_obst):
            if n_col_cnstr[n]>0:
                nu = Theta_obst_inv2[ob] @ ((C[0][3*K*i:3*K*(i+1)]) @ P_prev - np.tile((Obstacles[ob][0]), K).reshape(-1,1))
                Nu = np.zeros((n_col_cnstr[n],3*K))
                q =0
                xi_ij = np.zeros(n_col_cnstr[n],)
                for k in range(0,K):
                    if xi[k,n] < safe_dist:
                        Nu[q,3*k:3*(k+1)] = nu[3*k:3*(k+1)].flatten()
                        xi_ij[q]=xi[k,n]
                        q+=1
                Ain_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1]),:] = -Nu @ C[0][3*(K)*i:3*(K)*(i+1)]
                Ain_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1]),n_var-n_epsilon+n] = xi_ij
                bin_col[np.sum(n_col_cnstr[0:n]):np.sum(n_col_cnstr[0:n+1])] = -r_min*xi_ij - xi_ij**2 - (Nu @ C[0][3*(K)*i:3*(K)*(i+1)] @ P_prev).flatten()

                Ain_col[np.sum(n_col_cnstr)+n,n_var-n_epsilon+n] = 1
                bin_col[np.sum(n_col_cnstr)+n] = 0
            n +=1

    Ain = np.concatenate((Ain_der, Ain_col))
    bin = np.concatenate((bin_der, bin_col))

    Aeq = np.concatenate((Aeq_ini, Aeq_cont))
    beq = np.concatenate((beq_ini, beq_cont))
    J_line = J_line + np.dot(Pd.T,QC).reshape(-1)
     
    P_qp = cvxopt.matrix(J_quad)
    q_qp = cvxopt.matrix(J_line)
    G_qp = cvxopt.matrix(Ain)
    h_qp = cvxopt.matrix(bin)
    A_qp = cvxopt.matrix(Aeq)
    b_qp = cvxopt.matrix(beq)
    P_prev = cvxopt.matrix(P_prev)

    start = time.time()
    P = cvxopt.solvers.qp(P_qp,q_qp, G_qp, h_qp, A_qp, b_qp, kktsolver='ldl', options={'kktreg':1e-9, 'maxiters':20, 'show_progress':False}, initvals=P_prev)['x']
    #print("solver :", time.time()-start)
    ControlPoints = np.zeros((num_agent,L,3, n_order+1))
    for i in range(0,num_agent):
        for l in range(0,L):
            ControlPoints[i,l] = np.reshape(P[3*(i*(L*n_order+1) + l*n_order):3*(i*(L*n_order+1) + (l+1)*n_order+1)],((3,n_order+1)), order='F')
        
    return ControlPoints, P

def DMPC_update(X0, pd, dt_upd, dt_mpc, Tmax, Obstacles, param, constraints, speed=1):
    n_order, n_derivative, dt_opt, T_hor, L, theta, r_min, safe_dist, kappa, q, weight_quad, weight_line = param
    viz = False
    color = ['r','b','y','g','c','m','k']
    
    num_agent = X0.shape[0]
    Xc = X0
    K_bezier = int(T_hor/dt_upd)+1
    K_max = int(Tmax/dt_upd)
    n_MPC = int(Tmax/dt_mpc)-1
    n_MPC_update = 0

    Path_full = np.zeros((num_agent, n_derivative+1, 3, (K_max)))
    Path_horizon = np.zeros((num_agent, n_derivative+1, 3, L*(K_bezier)))
    Path_horizon_all = np.zeros((n_MPC, num_agent, 3, L*(K_bezier)))
    P = None
    qp_param = initDMPC_pos(num_agent, Obstacles, param, constraints, k_delay=0)
    for iter in range(1):
        ControlPoints, P = computeDMPC_pos(Xc, pd, Obstacles, param, qp_param, constraints, P)
    for i in range(0,num_agent): # agen
        for l in range(0,L):
            Path_horizon[i,:,:,l*K_bezier:(l+1)*K_bezier] = computeBezierCurve(ControlPoints[i][l], T_hor, dt_upd, n_derivative=n_derivative)[:]
    Path_horizon_all[n_MPC_update, :,:,:] = Path_horizon[:, 0, :,:]
    Xc = Path_horizon[:, :, :,0]
    Path_full[:,:,:,0] = Path_horizon[:, :, :,0]

    # Vizualisation
    plt.ion()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    angle=0
    
    t_reset = 0
    k_reset = 0
    t = 0
    k = 0
    print('MPC progress')
    pbar = tqdm(total=n_MPC-1)
    start = time.time()
    while k < K_max-1:
        t += dt_upd
        t_reset += dt_upd
        k +=1
        k_reset +=1
        
        if t_reset > dt_mpc:
            pbar.update(1)
            n_MPC_update = min(n_MPC_update+1,n_MPC-1)
            ControlPoints, P = computeDMPC_pos(Xc, pd, Obstacles, param, qp_param, constraints, P)
            for i in range(0,num_agent): # agen
                for l in range(0,ControlPoints[i].shape[0]):
                    Path_horizon[i,:,:,l*K_bezier:(l+1)*K_bezier] = computeBezierCurve(ControlPoints[i][l], T_hor, dt_upd, n_derivative=n_derivative)[:]
            Path_horizon_all[n_MPC_update, :,:,:] = Path_horizon[:, 0, :,:]
            t_reset = dt_upd
            k_reset = 1
        Xc = Path_horizon[:, :, :,k_reset]
            
        Path_full[:,:,:,k] = Xc[:,:,:]

        # Vizualisation
        if viz:
            ax.cla()
            ax.set_xlim(-2.5,2.5)
            ax.set_ylim(-2.5,2.5)
            ax.set_zlim(0,3)
            if Obstacles.shape[0] > 0 :
                for obstacle in Obstacles:
                    x = obstacle[0,0] + obstacle[1,0] * np.outer(np.cos(u), np.sin(v))
                    y = obstacle[0,1] + obstacle[1,1] * np.outer(np.sin(u), np.sin(v))
                    z = obstacle[0,2] + obstacle[1,2] * np.outer(np.ones_like(u), np.cos(v))
                    ax.plot_surface(x, y, z, color='y', alpha=0.4)
            for i in range(num_agent):
                ax.plot3D([pd[i,0]], [pd[i,1]], [pd[i,2]], 'go')
                ax.plot3D(Path_full[i,0][0,:k], Path_full[i,0][1,:k], Path_full[i,0][2,:k], color[i%7])
                ax.plot3D(Path_horizon[i,0,0,k_reset:], Path_horizon[i,0,1,k_reset:], Path_horizon[i,0,2,k_reset:], color[i%7]+'--')
                x = Path_full[i,0,0,k] + r_min*theta[0,0] * np.outer(np.cos(u), np.sin(v))
                y = Path_full[i,0,1,k] + r_min*theta[1,1] * np.outer(np.sin(u), np.sin(v))
                z = Path_full[i,0,2,k] + r_min*theta[2,2] * np.outer(np.ones_like(u), np.cos(v))
                ax.plot_surface(x, y, z, color=color[i%7], alpha=0.4)
            ax.view_init(elev=45*np.sin(angle*np.pi/180)+20, azim=angle)
            angle+=1
            plt.draw()
            plt.pause(dt_upd)

    pbar.close()
    print("Total MPC time : ", time.time()-start)
    np.save(Path(__file__).parent /'path.npy', Path_full)
    plotRealTime(Path_full[:,0,:,:], pd, Obstacles, theta*r_min, dt_upd, Path_horizon_all = Path_horizon_all, dt_mpc=dt_mpc, speed = 0.5, loop=True)

    return Path_full


def main(): 
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

    env_drone  = param["env_drone"]
    f_safe = param["f_safe"]
    r_min = (env_drone[0]**2+env_drone[1]**2+env_drone[2]**2)**0.5
    r_safe = f_safe*r_min
    theta = np.array([[env_drone[0]/r_min,0,0], [0,env_drone[1]/r_min,0], [0,0,env_drone[2]/r_min]])

    pos_static_obs = param["pos_static_obs"]
    dim_static_obs = param["dim_static_obs"]
    radius = param["radius"]
    height = param["height"]
    init_drone = param["init_drone"]
    goal_drone = param["goal_drone"]

    kappa = param["kappa"]
    weight_goal = param["weight_goal"]
    weight_quad = param["weight_quad"]
    weight_line = param["weight_line"]

    weight_vel = param["weight_vel"]
    weight_acc = param["weight_acc"]
    weight_jrk = param["weight_jrk"]

    x_lim  = param["x_lim"]
    y_lim  = param["y_lim"]
    z_lim  = param["z_lim"]
    vel_max = ((param["vel_max"])**2/3)**0.5
    acc_max = ((param["acc_max"])**2/3)**0.5

    obstacles = np.zeros((num_obs,2,3))
    for i in range(num_obs):
        obstacles[i,0] = pos_static_obs[i]
        obstacles[i,1] = dim_static_obs[i]
    
    N_curves = 1

    constraints = [[[0,1,2], np.array([[x_lim[0],y_lim[0],z_lim[0], x_lim[1],y_lim[1],z_lim[1]],[-vel_max,-vel_max, -vel_max, vel_max,vel_max,vel_max] ,[-acc_max,-acc_max,-acc_max, acc_max,acc_max,acc_max]])], np.array([weight_vel,weight_acc,weight_jrk,0,0])]    
    #param   n_order  n_derivative  dt_opt  T_hor  N_curves  Theta  r_min  r_safe   kappa  q  weight_quad weight_line
    param = [n_order, n_derivative, dt_opt, T_hor, N_curves, theta, r_min, r_safe, kappa, weight_goal, weight_quad, weight_line]
    
    initial_states = np.zeros((num_drone,2,3))
    target_states = np.zeros((num_drone,3))



    if (config == 'circle'):
        for i in range(num_drone):
            thetai = i* 2*np.pi/num_drone
            if (num_drone ==1):
                thetaf = np.pi
            else:
                thetaf = (i+int(num_drone/2))* 2*np.pi/num_drone
            initial_states[i,0] = np.array([radius * np.cos(thetai),radius * np.sin(thetai), height])
            target_states[i] = np.array([radius * np.cos(thetaf),radius * np.sin(thetaf), height])

    if (config == 'custom'):
        for i in range(num_drone):
            initial_states[i,0] = [init_drone[i][0], init_drone[i][1], init_drone[i][2]]
            target_states[i] = [goal_drone[i][0], goal_drone[i][1], goal_drone[i][2]]


    if (config == 'random'):
        x_range = [-2.5,2.5]
        y_range = [-2.5,2.5]
        z_range = [1,2]
        safe_dist = 0.75
        for i in range(num_drone):
            x = np.random.rand(2)*(x_range[1]-x_range[0]) + x_range[0]
            y = np.random.rand(2)*(y_range[1]-y_range[0]) + y_range[0]
            z = np.random.rand(2)*(z_range[1]-z_range[0]) + z_range[0]
            initial_states[i,0] = np.array([x[0],y[0], z[0]])
            target_states[i] = np.array([x[1],y[1], z[1]])
            for j in range(i):
                dist = min(np.linalg.norm(initial_states[j,0] - initial_states[i,0]),np.linalg.norm(target_states[j]-target_states[i]))
                while (dist<safe_dist):
                    x = np.random.rand(2)*(x_range[1]-x_range[0]) + x_range[0]
                    y = np.random.rand(2)*(y_range[1]-y_range[0]) + y_range[0]
                    z = np.random.rand(2)*(z_range[1]-z_range[0]) + z_range[0]
                    initial_states[i,0] = np.array([x[0],y[0], z[0]])
                    target_states[i] = np.array([x[1],y[1], z[1]])
                    dist = min(np.linalg.norm(initial_states[j,0] - initial_states[i,0]),np.linalg.norm(target_states[j]-target_states[i]))
    tfile = open(Path(__file__).parent/'config.txt', 'w')
    for i in range(num_drone):
        if i%2 :
            radio = 0
            freq= 85
        else:
            radio = 0
            freq = 85
        tfile.write('   cf' + str(i+1) + ': \n      enabled: true \n      uri: radio://'+ str(radio) + '/' + str(freq)+ '/2M/E7E7E7E70' + str(i+1) + '\n      initial_position: [' + str(round(initial_states[i,0,0],2)) + ', ' + str(round(initial_states[i,0,1], 2))+ ', 0]\n      type: cf21 \n')

            
    print('Initial states')
    print(initial_states[:,0])
    print('Target states')
    print(target_states)


    Path_full = DMPC_update(initial_states, target_states, dt_upd, dt_mpc, T_max, obstacles, param, constraints, speed=0.5)
    

    


if __name__ == "__main__":
    main()

    

// Author: Charles Sol
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <eigen-quadprog/QuadProg.h>


using namespace std;

double computeBinominal(int n, int k)
{
    double value = 1.0;
    for (int i = 1; i <= k; i++)
    {
        value = value * double(n + 1 - i) / double(i);
    }
    if (n == k){
        value = 1;
    }
    return value;
}

Eigen::MatrixXd computeBezierCoefficient(double t, double T, int n_order, int n_derivative)
{
    Eigen::MatrixXd bezierCoef = Eigen::MatrixXd::Zero(n_derivative+1,n_order+1);
    double Coef;
    for (int i = 0 ; i <= n_order; i+=1)
    {
        for (int d = 0 ; d <= n_derivative; d+=1)
        {
            if (i <= n_order-d)
            {
                Coef = computeBinominal(n_order-d, i) * std::pow((1 - t/T),double(n_order -d- i)) * std::pow((t/T),double(i)) * std::pow((1/T),double(d));
                for (int k = n_order-d+1 ; k <= n_order; k+=1)
                {
                    Coef *= double(k);
                }
                for (int j = 0 ; j <= d; j+=1)
                {
                    bezierCoef(d,j+i) += Coef * computeBinominal(d, j)* std::pow((-1),double(j+d));
                }
            }
        }

    }
    return bezierCoef; // Columns are coefficient of each control point and rows for each derivative
}

Eigen::MatrixXd** computeBezierCurve(Eigen::MatrixXd P, double T, double dt_upd, int n_derivative)
{
    int K = int (T/dt_upd);
    T = double (K) * dt_upd;
    double t;
    Eigen::MatrixXd bezierCoef;
    Eigen::MatrixXd** Curves_pointer;
    Curves_pointer = new Eigen::MatrixXd* [n_derivative+1];
    for (int k = 0 ; k <= K; k+=1)
    {
        t = double(k)*dt_upd;
        bezierCoef = computeBezierCoefficient(t, T, int(P.cols())-1,n_derivative);
        for (int d = 0 ; d <= n_derivative; d+=1)
        {
            if (k == 0)
            {
                Curves_pointer[d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,K+1));
            }
            Eigen::VectorXd point = P * bezierCoef.row(d).transpose();
            (*Curves_pointer[d]).col(k) = point;
        }
    }
    return Curves_pointer;
}

//-----------------------------------------------------------

tuple<Eigen::MatrixXd**, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd, Eigen::VectorXd**, Eigen::Matrix3d, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::Matrix3d**, Eigen::Matrix3d**, Eigen::MatrixXd**> initDMPC_pos(Eigen::MatrixXd** Obstacles, Eigen::Matrix3d theta, tuple <int, int, int, int, double, double, double, double, int, double, double, double> param, Eigen::MatrixXd** constraints)
{
    int num_agent; int num_obst; int n_order; int n_derivative; double dt_opt; double T_hor; double r_min; double safe_dist; int kappa; double weight_goal; double weight_quad; double weight_line;
    tie(num_agent, num_obst, n_order, n_derivative, dt_opt, T_hor, r_min, safe_dist, kappa, weight_goal, weight_quad, weight_line) = param;

    int K = int(T_hor/dt_opt);
    T_hor = double(K)*dt_opt;
    int n_epsilon = num_agent*num_obst + num_agent*(num_agent-1);
    int n_cstr_der = int((*constraints[0]).rows());

    int n_var = 3*num_agent*((n_order)+1)+n_epsilon;

    Eigen::MatrixXd** C; 
    C = new Eigen::MatrixXd* [n_derivative+1];
    for (int d = 0 ; d <= n_derivative; d+=1)
    {
        C[d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3*K*num_agent,n_var));
    }
    
    
    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3*num_agent*(K),3*num_agent*(K));

    Eigen::VectorXd** obs_tile; 
    obs_tile = new Eigen::VectorXd* [num_obst];
    for (int k=0; k< K; k+=1)
    {
        for (int ob = 0 ; ob < num_obst; ob+=1)
        {
            if (k==0)
            {
                obs_tile[ob] = new (Eigen::VectorXd)(Eigen::VectorXd::Zero(3*K));
            }
            (*obs_tile[ob]).block(3*k,0,3,1) = (*Obstacles[0]).row(ob).transpose();
        }
    }
       
        
    Eigen::MatrixXd coef_end_point = computeBezierCoefficient(T_hor, T_hor, n_order, n_derivative);
    
    Eigen::Matrix3d theta_quad_inv = theta.inverse();
    Eigen::MatrixXd theta_quad_inv2 = theta_quad_inv*theta_quad_inv;
    Eigen::MatrixXd Theta_quad_inv2 = Eigen::MatrixXd::Zero(3*K,3*K);
    Eigen::Matrix3d** theta_obst_inv;
    Eigen::Matrix3d** theta_obst_inv2; 
    Eigen::MatrixXd** Theta_obst_inv2; 

    Eigen::Matrix3d theta_ob;

    theta_obst_inv = new Eigen::Matrix3d* [num_obst];
    theta_obst_inv2 = new Eigen::Matrix3d* [num_obst];
    Theta_obst_inv2 = new Eigen::MatrixXd* [num_obst];
    for (int i = 0 ; i < num_obst; i+=1)
    {
        theta_ob << ((*Obstacles[1])(i,0)/r_min+theta(0,0))/2,0,0,   0, ((*Obstacles[1])(i,1)/r_min+theta(1,1))/2, 0,   0,0,((*Obstacles[1])(i,2)/r_min + theta(2,2))/2;
        theta_obst_inv[i] = new (Eigen::Matrix3d)(theta_ob.inverse());
        theta_obst_inv2[i] = new (Eigen::Matrix3d)((theta_ob.inverse())*(theta_ob.inverse()));
        Theta_obst_inv2[i] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3*K,3*K));
    }

    Eigen::MatrixXd Ain = Eigen::MatrixXd::Zero(3*K*num_agent*(2*n_cstr_der),n_var);
    Eigen::VectorXd bin = Eigen::VectorXd::Zero(3*K*num_agent*(2*n_cstr_der));

    Eigen::MatrixXd Aeq = Eigen::MatrixXd::Zero(3*(num_agent*(n_derivative+1)),n_var);
    Eigen::VectorXd beq = Eigen::VectorXd::Zero(3*(num_agent*(n_derivative+1)));


    double t; Eigen::MatrixXd coef;
    for (int k=0; k<K; k+=1)
    {
        Theta_quad_inv2.block(3*k,3*k,3, 3) = theta_quad_inv2;
        for (int ob=0; ob<num_obst; ob+=1)
        {
            (*Theta_obst_inv2[ob]).block(3*k,3*k,3,3) = *theta_obst_inv2[ob];
        }
        t = double(k%K)*dt_opt;
        coef = computeBezierCoefficient(t, T_hor, n_order, n_derivative);
        for (int i=0; i < num_agent; i+=1)
        {
            if (k>K-1-kappa)
            {
                Q.block(3*(K*i+k),3*(K*i+k),3,3) << weight_goal,0,0,0,weight_goal,0,0,0,weight_goal;
            }
            for (int n=0; n<=n_order; n+=1)
            {
                for (int m=0; m<n_cstr_der; m+=1)
                {
                    Ain.block(3*(i*(2*(K*n_cstr_der)) + k*2*n_cstr_der+2*m),3*(i*(n_order+1)+n),3,3) << -coef(m,n),0,0, 0,-coef(m,n),0, 0,0,-coef(m,n);
                    bin.block(3*(i*(2*(K*n_cstr_der)) + k*2*n_cstr_der+2*m),0,3,1) << -(*constraints[0])(m,0), -(*constraints[0])(m,1), -(*constraints[0])(m,2);
                    Ain.block(3*(i*(2*(K*n_cstr_der)) + k*2*n_cstr_der+2*m+1),3*(i*(n_order+1)+n),3,3) << coef(m,n),0,0, 0,coef(m,n),0, 0,0,coef(m,n);
                    bin.block(3*(i*(2*(K*n_cstr_der)) + k*2*n_cstr_der+2*m+1),0,3,1) << (*constraints[0])(m,3), (*constraints[0])(m,4), (*constraints[0])(m,5);
                }
                for (int d=0; d<=n_derivative; d+=1)
                {
                    (*C[d]).block(3*(k+K*i),3*(i*(n_order+1)+n),3,3) << coef(d,n),0,0, 0,coef(d,n),0, 0,0,coef(d,n);
                }
            }
        }
                    
    }

    Eigen::MatrixXd J_quad = Eigen::MatrixXd::Zero(n_var, n_var);
    Eigen::RowVectorXd J_line = Eigen::RowVectorXd::Zero(n_var);
    Eigen::MatrixXd QC = Eigen::MatrixXd::Zero(3*K, n_var);
    for (int n=0; n<n_epsilon; n+=1)
    {
        J_quad(n_var-n_epsilon+n, n_var-n_epsilon+n)=weight_quad;
        J_line(n_var-n_epsilon+n) = -weight_line;
    }

    J_quad += (*C[0]).transpose() * Q * (*C[0]);
    QC = -(Q * (*C[0]));

    for (int d =1; d<n_derivative; d+=1)
    {
        J_quad += (*constraints[1])(0,d-1) * (*C[d]).transpose() * (*C[d]);
    }

    return make_tuple(C, Q, Ain, bin, Aeq, beq, J_quad, J_line, QC, obs_tile, theta_quad_inv, theta_quad_inv2, Theta_quad_inv2, theta_obst_inv, theta_obst_inv2, Theta_obst_inv2);
}

tuple<Eigen::MatrixXd**, Eigen::VectorXd> computeDMPC_pos(Eigen::MatrixXd** X0, Eigen::MatrixXd pd, Eigen::MatrixXd** Obstacles, tuple <int, int, int, int, double, double, double, double, int, double, double, double> param, tuple<Eigen::MatrixXd**, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd,  Eigen::VectorXd**, Eigen::Matrix3d, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::Matrix3d**, Eigen::Matrix3d**, Eigen::MatrixXd**> qp_Param, Eigen::MatrixXd** constraints, Eigen::VectorXd P_prev, Eigen::MatrixXd** Control_Point)
{
    Eigen::MatrixXd** C; Eigen::MatrixXd Q; Eigen::MatrixXd Ain_der; Eigen::VectorXd bin_der; Eigen::MatrixXd Aeq; Eigen::VectorXd beq; Eigen::MatrixXd J_quad; Eigen::RowVectorXd J_line; Eigen::MatrixXd QC; Eigen::VectorXd** obs_tile; Eigen::Matrix3d theta_quad_inv; Eigen::MatrixXd theta_quad_inv2; Eigen::MatrixXd Theta_quad_inv2; Eigen::Matrix3d** theta_obst_inv; Eigen::Matrix3d** theta_obst_inv2; Eigen::MatrixXd** Theta_obst_inv2;
    tie(C, Q, Ain_der, bin_der, Aeq, beq, J_quad, J_line, QC, obs_tile, theta_quad_inv, theta_quad_inv2, Theta_quad_inv2, theta_obst_inv, theta_obst_inv2, Theta_obst_inv2) = qp_Param;
    int num_agent; int num_obst; int n_order; int n_derivative; double dt_opt; double T_hor; double r_min; double safe_dist; int kappa; double weight_goal; double weight_quad; double weight_line;
    tie(num_agent, num_obst, n_order, n_derivative, dt_opt, T_hor, r_min, safe_dist, kappa, weight_goal, weight_quad, weight_line) = param;

    int K = int(T_hor/dt_opt);
    T_hor = double(K)*dt_opt;
    int n_epsilon = num_agent*num_obst + num_agent*(num_agent-1);
    int n_cstr_der = int((*constraints[0]).rows());

    int n_var = 3*num_agent*((n_order)+1)+n_epsilon;

    Eigen::VectorXd Pd = Eigen::VectorXd::Zero(3*K*num_agent);
    Eigen::VectorXd P(n_var);
    if (P_prev == P.setZero())
    {
        for (int i=0; i < num_agent; i+=1)
        {
            for (int j=0; j< (n_order+1); j+=1)
            {
                P_prev.block(3*(i*(n_order+1)+j),0,3,1) = (*X0[0]).row(i).transpose();
            }
        }
    }        
            
    Eigen::MatrixXd xi = Eigen::MatrixXd::Ones(K,num_agent*(num_agent-1) + num_agent*num_obst)*safe_dist;
    Eigen::VectorXi n_col_cnstr = Eigen::VectorXi::Zero(num_agent*(num_agent-1)+ num_agent*num_obst);
    auto start = std::chrono::system_clock::now();    
    double t; Eigen::MatrixXd coef; int r;
    for (int k=0; k<K; k+=1)
    {
        r=0;
        for (int i=0; i < num_agent; i+=1)
        {
            Pd.block(3*(K*i+k),0,3,1) = pd.row(i).transpose();
            if (k ==0)
            {
                t = double(k%K)*dt_opt;
                coef = computeBezierCoefficient(t, T_hor, n_order, n_derivative);
                for (int n=0; n<=n_order; n+=1)
                {
                
                    for (int d=0; d < (n_derivative+1); d+=1) // initial state constraints
                    {
                        Aeq.block(3*(i*(n_derivative+1)+d),3*(i*(n_order+1)+n),3,3) = coef(d,n)*(Eigen::Matrix3d::Identity());
                        beq.block(3*(i*(n_derivative+1)+d),0,3,1) = (*X0[d]).row(i).transpose();
                    }       
                }
            }
            for (int j=0; j < num_agent; j+=1)
            {
                if (i != j)
                {
                    xi(k,r)= (theta_quad_inv * ((*C[0]).block(3*(K*i+k),0,3,n_var)-(*C[0]).block(3*(K*j+k),0,3,n_var)) * P_prev.block(0,0,n_var,1) ).norm();
                    if (xi(k,r) < safe_dist)
                    {
                        n_col_cnstr(r) += 1;
                    }  
                    r +=1;
                }
            }
        }
        if (num_obst > 0) 
        {
            for (int i=0; i < num_agent; i+=1)
            {
                for (int ob=0; ob < num_obst; ob+=1)
                {
                    xi(k,r)= ((*theta_obst_inv[ob]) * ((*C[0]).block(3*(K*i+k),0,3,n_var) * P_prev.block(0,0,n_var,1)- ((*Obstacles[0]).row(ob)).transpose()) ).norm();
                    if (xi(k,r) < safe_dist) 
                    {
                        n_col_cnstr(r) += 1;
                    }
                    r +=1;
                }
            }
        }             
    }
    int n = 0; int q; Eigen::VectorXd nu; Eigen::MatrixXd Nu; Eigen::VectorXd xi_ij;
    Eigen::MatrixXd Ain_col = Eigen::MatrixXd::Zero(n_col_cnstr.sum()+n_epsilon,n_var);
    Eigen::VectorXd bin_col = Eigen::VectorXd::Zero(n_col_cnstr.sum()+n_epsilon);
    
    auto end = std::chrono::system_clock::now();  
    std::chrono::duration<double> elapsed_seconds = end-start;
    //cout << " -- part1 : " << elapsed_seconds.count() << 's';
    for (int i=0; i < num_agent; i+=1)
    {
        for (int j=0; j < num_agent; j+=1)
        {
            if (i != j)
            {
                if (n_col_cnstr(n)>0)
                {
                    nu = Theta_quad_inv2 * ((*C[0]).block(3*(K*i),0,3*K,n_var)-(*C[0]).block(3*(K*j),0,3*K,n_var)) * P_prev.block(0,0,n_var,1);
                    Nu = Eigen::MatrixXd::Zero(n_col_cnstr(n),3*K);
                    q=0;
                    xi_ij = Eigen::VectorXd::Zero(n_col_cnstr(n));
                    for (int k=0; k<K; k+=1)
                    {
                        if (xi(k,n) < safe_dist)
                        {
                            Nu.block(q,3*k,1,3) = nu.block(3*k,0,3,1).transpose();
                            xi_ij(q)=xi(k,n);
                            q+=1;
                        }
                    }
                    Ain_col.block(n_col_cnstr.block(0,0,n,1).sum(),0,n_col_cnstr(n),n_var) = -Nu * (*C[0]).block(3*(K*i),0,3*K,n_var);
                    Ain_col.block(n_col_cnstr.block(0,0,n,1).sum(),n_var-n_epsilon+n,n_col_cnstr(n),1) = xi_ij;
                    bin_col.block(n_col_cnstr.block(0,0,n,1).sum(),0,n_col_cnstr(n),1) = -r_min * xi_ij - Eigen::VectorXd(xi_ij.array().pow(2)) - (Nu * (*C[0]).block(3*(K*i),0,3*K,n_var) * P_prev.block(0,0,n_var,1));

                    Ain_col(n_col_cnstr.sum()+n,n_var-n_epsilon+n) = 1;
                    bin_col(n_col_cnstr.sum()+n) = 0;
                }
                n +=1;
            }
        }
    }
    if (num_obst > 0) 
    {
        for (int i=0; i < num_agent; i+=1)
        {
            for (int ob=0; ob < num_obst; ob+=1)
            {
                if (n_col_cnstr(n)>0)
                {
                    nu = (*Theta_obst_inv2[ob]) * ((*C[0]).block(3*(K*i),0,3*K,n_var) * P_prev.block(0,0,n_var,1) -(*obs_tile[ob]));
                    Nu = Eigen::MatrixXd::Zero(n_col_cnstr(n),3*K);
                    q=0;
                    xi_ij = Eigen::VectorXd::Zero(n_col_cnstr(n));
                    for (int k=0; k<K; k+=1)
                    {
                        if (xi(k,n) < safe_dist)
                        {
                            Nu.block(q,3*k,1,3) = nu.block(3*k,0,3,1).transpose();
                            xi_ij(q)=xi(k,n);
                            q+=1;
                        }
                    }
                    Ain_col.block(n_col_cnstr.block(0,0,n,1).sum(),0,n_col_cnstr(n),n_var) = -Nu * (*C[0]).block(3*(K*i),0,3*K,n_var);
                    Ain_col.block(n_col_cnstr.block(0,0,n,1).sum(),n_var-n_epsilon+n,n_col_cnstr(n),1) = xi_ij;
                    bin_col.block(n_col_cnstr.block(0,0,n,1).sum(),0,n_col_cnstr(n),1) = -r_min * xi_ij - Eigen::VectorXd(xi_ij.array().pow(2)) - (Nu * (*C[0]).block(3*(K*i),0,3*K,n_var) * P_prev.block(0,0,n_var,1));

                    Ain_col(n_col_cnstr.sum()+n,n_var-n_epsilon+n) = 1;
                    bin_col(n_col_cnstr.sum()+n) = 0;
                }
                n +=1;
            }
        }
    }
    
    Eigen::MatrixXd Ain = Eigen::MatrixXd::Zero(3*K*num_agent*(2*n_cstr_der)+n_col_cnstr.sum()+n_epsilon,n_var);
    Eigen::VectorXd bin = Eigen::VectorXd::Zero(3*K*num_agent*(2*n_cstr_der)+n_col_cnstr.sum()+n_epsilon);
    Ain.block(0,0,3*K*num_agent*(2*n_cstr_der),n_var) = Ain_der;
    bin.block(0,0,3*K*num_agent*(2*n_cstr_der),1) = bin_der;
    Ain.block(3*K*num_agent*(2*n_cstr_der),0,n_col_cnstr.sum()+n_epsilon,n_var) = Ain_col;
    bin.block(3*K*num_agent*(2*n_cstr_der),0,n_col_cnstr.sum()+n_epsilon,1) = bin_col;

    J_line += (Pd.transpose() * QC);

    start = std::chrono::system_clock::now();    
    Eigen::QuadProgDense qp_solver(n_var, int(Aeq.rows()), int(Ain.rows()));
	qp_solver.solve(J_quad, J_line, Aeq, beq, Ain, bin);
    end = std::chrono::system_clock::now();  
    elapsed_seconds = end-start;
    cout << " -- qp time : " << elapsed_seconds.count() << 's';

    P = qp_solver.result();
    
    for (int i=0; i < num_agent; i+=1)
    {
        for (int n = 0 ; n < n_order+1; n+=1)
        {
            (*Control_Point[i]).col(n) = P.block(3*(i*(n_order+1)) + 3*n,0,3,1);
        }
    }


    return make_tuple(Control_Point, P);
}

tuple<Eigen::MatrixXd***, Eigen::MatrixXd***> DMPC_update(Eigen::MatrixXd** X0, Eigen::MatrixXd pd, Eigen::MatrixXd** Ob, Eigen::Matrix3d theta, double dt_upd, double dt_mpc, double Tmax, tuple <int, int, int, int, double, double, double, double, int, double, double, double> Param, Eigen::MatrixXd** Constraints)
{
    int num_agent; int num_obst; int n_order; int n_derivative; double dt_opt; double T_hor; double r_min; double safe_dist; int kappa; double weight_goal; double weight_quad; double weight_line;
    tie(num_agent, num_obst, n_order, n_derivative, dt_opt, T_hor, r_min, safe_dist, kappa, weight_goal, weight_quad, weight_line) = Param;

    int K = int(T_hor/dt_upd);
    int K_max = int(Tmax/dt_upd);
    int n_MPC = int(Tmax/dt_mpc);
    int n_MPC_update = 0;

    T_hor = double(int(T_hor/dt_upd))*dt_upd;
    int n_epsilon = num_agent*num_obst + num_agent*(num_agent-1);
    int n_var = 3*num_agent*((n_order)+1)+n_epsilon;

    Eigen::MatrixXd*** Path_full;
    Eigen::MatrixXd*** Path_horizon;
    Eigen::MatrixXd*** Path_horizon_all;
    Eigen::MatrixXd** Control_Point;
    Path_full = new Eigen::MatrixXd** [num_agent];
    Path_horizon = new Eigen::MatrixXd** [num_agent];
    Path_horizon_all = new Eigen::MatrixXd** [n_MPC];
    Control_Point = new Eigen::MatrixXd* [num_agent];
    for (int i = 0 ; i < num_agent; i+=1)
    {
        Path_full[i] = new Eigen::MatrixXd* [n_derivative+1];
        Path_horizon[i] = new Eigen::MatrixXd* [n_derivative+1];
        for (int d = 0 ; d < n_derivative+1; d+=1)
        {
            Path_full[i][d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,K_max));
            Path_horizon[i][d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,(K+1)));
        }
        Control_Point[i] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,n_order+1));
        
    }
    for (int n = 0 ; n < n_MPC; n+=1)
    {
        Path_horizon_all[n] = new Eigen::MatrixXd* [num_agent];
        for (int i = 0 ; i < num_agent; i+=1)
        {
            Path_horizon_all[n][i] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,(K)));
        }
    }
    Eigen::VectorXd P(n_var);

    P = P.setZero();
    Eigen::MatrixXd** curves;

    tuple<Eigen::MatrixXd**, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::RowVectorXd, Eigen::MatrixXd, Eigen::VectorXd**, Eigen::Matrix3d, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::Matrix3d**, Eigen::Matrix3d**, Eigen::MatrixXd**> qp_Param;
    qp_Param = initDMPC_pos(Ob, theta, Param, Constraints);

    auto start = std::chrono::system_clock::now();    
    tie(Control_Point, P) =  computeDMPC_pos(X0, pd, Ob, Param, qp_Param, Constraints, P, Control_Point);
    auto end = std::chrono::system_clock::now();
    for (int i = 0 ; i < num_agent; i+=1)
    {       
            curves = computeBezierCurve((*Control_Point[i]), T_hor, dt_upd, n_derivative);
            for (int d = 0 ; d < n_derivative+1; d+=1)
            {
                (*Path_horizon[i][d]) = (*curves[d]);
                (*X0[d]).row(i) = ((*curves[d]).col(0)).transpose();
                (*Path_full[i][d]).col(0) = (*curves[d]).col(0);
            }
            (*Path_horizon_all[n_MPC_update][i]) = (*curves[0]);
    }
    double t_reset = 0;
    int k_reset = 0;
    double t = 0;
    int k = 0;
    double radius = 2.4;
    double height = 1.5;
    double speed_turn = 0.5;
    double offset =  Tmax/3;
    while (k < K_max-1)
    {
        t += dt_upd;
        t_reset += dt_upd;
        k +=1;
        k_reset +=1;
        if (t_reset > dt_mpc)
        {
            n_MPC_update = min(n_MPC_update+1,n_MPC-1);
            start = std::chrono::system_clock::now();  
            tie(Control_Point, P) =  computeDMPC_pos(X0, pd, Ob, Param, qp_Param, Constraints,  P, Control_Point);
            for (int i = 0 ; i < num_agent; i+=1)
            {
                    curves = computeBezierCurve((*Control_Point[i]), T_hor, dt_upd, n_derivative);
                    for (int d = 0 ; d < n_derivative+1; d+=1)
                    {
                        (*Path_horizon[i][d]) = (*curves[d]);
                    }
                    (*Path_horizon_all[n_MPC_update][i]) = (*curves[0]);

            }
            t_reset = dt_upd;
            k_reset = 1;
            end = std::chrono::system_clock::now();  
            std::chrono::duration<double> elapsed_seconds = end-start;
            
            cout << '\r' << "progression : " << (k*100/(K_max-1)) << "% " << "iteration time : " << elapsed_seconds.count() << 's';
            cout.flush();
        }
        for (int i = 0 ; i < num_agent; i+=1)
        {
                for (int d = 0 ; d < n_derivative+1; d+=1)
                {
                    (*X0[d]).row(i) = (*Path_horizon[i][d]).col(k_reset).transpose();
                    (*Path_full[i][d]).col(k) = (*Path_horizon[i][d]).col(k_reset);
                }
        }
        
        if (false) {
            for (int i = 0 ; i < num_agent; i+=1)
            {
                if (t > Tmax/3)
                {
                    if (t < 2*Tmax/3)
                    {
                        pd.row(i) << radius * cos((t-offset)*speed_turn+(i+int(num_agent/2))* 2*M_PI/num_agent),radius * sin((t-offset)*speed_turn+(i+int(num_agent/2))* 2*M_PI/num_agent), height;   
                    }
                    else{
                        pd.row(i) << radius * cos(i* 2*M_PI/num_agent),radius * sin(i* 2*M_PI/num_agent), height;   
                    }
                }
            
            }
        }
        
            
    }

    Eigen::MatrixXd** C; Eigen::MatrixXd Q; Eigen::MatrixXd Ain_der; Eigen::VectorXd bin_der; Eigen::MatrixXd Aeq; Eigen::VectorXd beq; Eigen::MatrixXd J_quad; Eigen::RowVectorXd J_line; Eigen::MatrixXd QC; Eigen::VectorXd** obs_tile; Eigen::Matrix3d theta_quad_inv; Eigen::MatrixXd theta_quad_inv2; Eigen::MatrixXd Theta_quad_inv2; Eigen::Matrix3d** theta_obst_inv; Eigen::Matrix3d** theta_obst_inv2; Eigen::MatrixXd** Theta_obst_inv2;
    tie(C, Q, Ain_der, bin_der, Aeq, beq, J_quad, J_line, QC, obs_tile, theta_quad_inv, theta_quad_inv2, Theta_quad_inv2, theta_obst_inv, theta_obst_inv2, Theta_obst_inv2) = qp_Param;

    for (int d = 0 ; d <= n_derivative; d+=1)
    {
        delete C[d];
    }
    for (int ob = 0 ; ob < num_obst; ob+=1)
    {
        delete obs_tile[ob];
        delete theta_obst_inv[ob];
        delete theta_obst_inv2[ob];
        delete Theta_obst_inv2[ob];
    }
    delete C;
    delete obs_tile;
    delete theta_obst_inv;
    delete theta_obst_inv2;
    delete Theta_obst_inv2;
    
    for (int i = 0 ; i < num_agent; i+=1)
    {
        for (int d = 0 ; d < n_derivative+1; d+=1)
        {
            delete Path_horizon[i][d];
        }
    }
    for (int i = 0 ; i < num_agent; i+=1)
    {
        delete Path_horizon[i];
        delete Control_Point[i];
    }
    for (int d = 0 ; d <= n_derivative; d+=1)
    {
        delete curves[d];
    }
    delete Path_horizon;
    delete Control_Point;
    delete curves;

    return make_tuple(Path_full, Path_horizon_all);
    
}

int main()
{   
    YAML::Node param = YAML::LoadFile("ros2_ws/src/dmpc_swarm/dmpc_swarm/config_swarm.yaml");
    int num_drone = param["num_drone"].as<int>();
    int num_obs = param["num_obs"].as<int>();
    cout << "number of drones : " << num_drone << " number of obstacles : " << num_obs << endl;
    string config = param["config"].as<string>();

    int n_order = param["n_order"].as<int>();
    int n_derivative = param["n_derivative"].as<int>();
    double dt_upd = param["dt_upd"].as<double>();
    double dt_opt = param["dt_opt"].as<double>();
    double dt_mpc = param["dt_mpc"].as<double>();
    double T_hor = param["T_hor"].as<double>();
    double T_max = param["T_max"].as<double>();

    
    int kappa = param["kappa"].as<int>();
    double weight_goal = param["weight_goal"].as<double>();
    double weight_quad = param["weight_quad"].as<double>();
    double weight_line = param["weight_line"].as<double>();

    double weight_vel = param["weight_vel"].as<double>();
    double weight_acc = param["weight_acc"].as<double>();
    double weight_jrk = param["weight_jrk"].as<double>();

    std::vector<float> x_lim  = param["x_lim"].as<std::vector<float>>();
    std::vector<float> y_lim  = param["y_lim"].as<std::vector<float>>();
    std::vector<float> z_lim  = param["z_lim"].as<std::vector<float>>();
    double vel_max = pow(pow(param["vel_max"].as<double>(),2)/3,0.5);
    double acc_max = pow(pow(param["acc_max"].as<double>(),2)/3,0.5);

    std::vector<float> env_drone  = param["env_drone"].as<std::vector<float>>();
    double f_safe = param["f_safe"].as<double>();
    double r_min = pow(pow(env_drone[0],2)+pow(env_drone[1],2)+pow(env_drone[2],2),0.5);
    double safe_dist = f_safe*r_min;
    Eigen::Matrix3d theta;
    theta << env_drone[0]/r_min,0,0, 0,env_drone[1]/r_min,0, 0,0,env_drone[2]/r_min;

    Eigen::MatrixXd** Constraints;
    Constraints = new Eigen::MatrixXd* [2];
    Constraints[0] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(3,6));
    Constraints[1] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(1,n_derivative-1));
    (*Constraints[0]) << x_lim[0],y_lim[0],z_lim[0], x_lim[1],y_lim[1],z_lim[1], -vel_max,-vel_max,-vel_max, vel_max,vel_max,vel_max, -acc_max,-acc_max,-acc_max, acc_max,acc_max,acc_max;
    (*Constraints[1]).row(0) << weight_vel, weight_acc, weight_jrk;

    Eigen::MatrixXd** X0;
    Eigen::MatrixXd Pd(num_drone,3);
    X0 = new Eigen::MatrixXd* [n_derivative+1];
    for (int d = 0 ; d < n_derivative+1; d+=1)
    {
        X0[d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(num_drone,3));
    }
    Eigen::MatrixXd** Ob;
    Ob = new Eigen::MatrixXd* [2];
    for (int d = 0 ; d < 2; d+=1)
    {
        Ob[d] = new (Eigen::MatrixXd)(Eigen::MatrixXd::Zero(num_obs,3));
    }

    std :: vector<std :: vector<double>> pos_static_obs = param["pos_static_obs"].as<std :: vector<std :: vector<double>> >();
	std :: vector<std :: vector<double>> dim_static_obs = param["dim_static_obs"].as<std :: vector<std :: vector<double>> >();

    if (config.compare("circle")==0)
    {
        cout << "circle configuration" << endl;
        double radius = param["radius"].as<double>();
        double height = param["height"].as<double>();
        for (int i = 0 ; i < num_drone; i+=1)
        {
            (*X0[0]).row(i) << radius * cos(i* 2*M_PI/num_drone), radius * sin(i* 2*M_PI/num_drone), height;
            if (num_drone==1)
            {
                Pd.row(i) << radius * cos( M_PI),radius * sin(M_PI), height;
            }
            else
            {
                Pd.row(i) << radius * cos((i+int(num_drone/2))* 2*M_PI/num_drone),radius * sin((i+int(num_drone/2))* 2*M_PI/num_drone), height;
            }  
        }
    }
    else if (config.compare("custom")==0)
    {
        cout << "custom configuration" << endl;
        std :: vector<std :: vector<double>> init_drone = param["init_drone"].as<std :: vector<std :: vector<double>> >();
		std :: vector<std :: vector<double>> goal_drone = param["goal_drone"].as<std :: vector<std :: vector<double>> >();
        for (int i = 0 ; i < num_drone; i+=1)
        {
            (*X0[0]).row(i) << init_drone[i][0], init_drone[i][1], init_drone[i][2];
            Pd.row(i) << goal_drone[i][0], goal_drone[i][1], goal_drone[i][2];
        }
    }

    for (int i = 0 ; i < num_obs; i+=1)
    {
        (*Ob[0]).row(i) << pos_static_obs[i][0], pos_static_obs[i][1], pos_static_obs[i][2];
        (*Ob[1]).row(i) << dim_static_obs[i][0], dim_static_obs[i][1], dim_static_obs[i][2];
    }

    int n_epsilon = num_drone * num_obs + num_drone * (num_drone-1);
    int n_var = 3*num_drone*((n_order)+1)+n_epsilon;

    Eigen::VectorXd P(n_var);
    P = P.setZero();

    tuple <int, int, int, int, double, double, double, double, int, double, double, double> Param;
    Param = make_tuple(num_drone, num_obs, n_order, n_derivative, dt_opt, T_hor, r_min, safe_dist, kappa, weight_goal, weight_quad, weight_line);
    Eigen::MatrixXd*** Path_full;
    Eigen::MatrixXd*** Path_horizon_all;
    tie(Path_full, Path_horizon_all) = DMPC_update(X0, Pd, Ob, theta, dt_upd, dt_mpc, T_max, Param, Constraints);
    ofstream file("ros2_ws/src/dmpc_swarm/dmpc_swarm/path.csv");
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    if (file.is_open())
    {
        file << fixed;
        file << setprecision(5);
        for (int i = 0 ; i < num_drone; i+=1)
        {
            for (int d = 0 ; d < n_derivative+1; d+=1)
            {
                file << (*Path_full[i][d]).format(CSVFormat) << endl;
            }
        }
        file.close();
    }
    ofstream file2("ros2_ws/src/dmpc_swarm/dmpc_swarm/path_horizon.csv");
    if (file2.is_open())
    {
        file2 << fixed;
        file2 << setprecision(5);
        int N_MPC = int(T_max/dt_mpc)-1;
        for (int n = 0 ; n < N_MPC; n+=1)
        {
            for (int i = 0 ; i < num_drone; i+=1)
            {
                file2 << (*Path_horizon_all[n][i]).format(CSVFormat) << endl;
            }
        }
        file2.close();
    }
    cout << endl << "path planning finished with success" << endl << endl;

}
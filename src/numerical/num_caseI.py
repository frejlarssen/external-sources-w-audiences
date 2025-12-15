import numpy as np
import src.numerical.num_general as numgen

def dims(params):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    n_1 = int(np.rint(alpha * n))
    n_2 = n-n_1
    print(n_1, n_2, k)
    return(n_1, n_2, k)

def sum_of_equilibrium(params):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    (n_1, n_2, k) = dims(params)
    
    print("(num.n_1, num.n_2, num.n, num.k): ", (n_1, n_2, n, k))

    mathbf_I = np.eye(n)
    mathbf_D = (n-1) * mathbf_I
    mathbf_W = np.ones((n, n)) - np.eye(n)
    mathbf_L = mathbf_D-mathbf_W
    
    
    bar_mathbf_s_M      = (  alpha+delta)/(   alpha *n) * e_top_s
    bar_mathbf_s_M_prim = (1-alpha-delta)/((1-alpha)*n) * e_top_s
    #Assume uniform distribution within each audience
    mathbf_s = np.block([
        bar_mathbf_s_M      * np.ones(n_1),
        bar_mathbf_s_M_prim * np.ones(n_2)
    ])

    bar_s_M      = (  alpha+delta)/( alpha   *n) * e_top_s
    bar_s_M_prim = (1-alpha-delta)/((1-alpha)*n) * e_top_s
    
    print("bar_s_M: ", bar_s_M)
    print("gamma: ", gamma)
    print("(1+gamma)*bar_s_M: ", (1+gamma)*bar_s_M)
    
    z_M      = np.minimum((1+gamma)*bar_s_M     , 1)
    
    print("z_M:", z_M)
    
    z_M_prim =        (1-gamma)*bar_s_M_prim
    
    zeta = np.block([
        z_M      * np.ones(n_1),
        z_M_prim * np.ones(n_2)
    ])
    
    print("num zeta:")
    print(zeta)

    z_equi = numgen.equilibrium(mathbf_I, mathbf_D, mathbf_L, mathbf_s, beta, zeta)
    
    print("num z_equi:")
    print(z_equi)
    
    e_M_top = np.transpose(
        np.block([
            np.ones(n_1),
            np.zeros(n_2)
        ])
    )
    e_M_prim_top = np.transpose(
        np.block([
            np.zeros(n_1),
            np.ones(n_2)
        ])
    )

    e_M_top_z_equi      = e_M_top      @ z_equi
    e_M_prim_top_z_equi = e_M_prim_top @ z_equi
    
    

    return (e_M_top_z_equi, e_M_prim_top_z_equi)

def avg_of_equilibrium_M(params):
    (n_1, n_2, k) = dims(params)
    sum_of_equilibrium(params)[0] / n_1

def avg_of_equilibrium_M_prim(params):
    (n_1, n_2, k) = dims(params)
    sum_of_equilibrium(params)[1] / n_2
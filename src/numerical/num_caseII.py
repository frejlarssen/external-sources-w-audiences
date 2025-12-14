import numpy as np

def naive_sum_equilibrium_expressed_opinions_non_trunc(params):
    (alpha, beta, gamma, delta, d, n, e_top_s) = params
    #z_M = 

def naive_dims(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    n_1 = int(np.rint(alpha * n))
    n_2 = n-n_1
    print(n_1, n_2, k)
    return(n_1, n_2, k)

def naive_mathbf_I_and_D(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    (n_1, n_2, k) = naive_dims(alpha, beta, gamma, delta, n, k, e_top_s, trunc)

    diag_as_vector = np.block([n_1*np.ones(k), (n_1-1)*np.ones(n_1-k), n_2*np.ones(k), (n_2-1)*np.ones(n_2-k)])
    
    mathbf_D = np.diag(
        diag_as_vector
    )

    mathbf_I = np.eye(n)
    
    return (mathbf_I, mathbf_D)

def naive_mathbf_M(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    
    (n_1, n_2, k) = naive_dims(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    #mathbf_W = np.array([ #TODO: Make general
    #    [0, 1, 1, 1, 0],
    #    [1, 0, 1, 0, 0],
    #    [1, 1, 0, 0, 0],
    #    [1, 0, 0, 0, 1],
    #    [0, 0, 0, 1, 0]
    #])
    
    mathbf_K = np.block([
        [np.eye(k), np.zeros((k,n_2-k))],
        [np.zeros((n_1-k,k)), np.zeros((n_1-k,n_2-k))]
    ])

    mathbf_W = np.block([
        [np.ones((n_1,n_1)), mathbf_K],
        [np.transpose(mathbf_K), np.ones((n_2,n_2))]
    ]) - np.eye(n)
    
    #mathbf_D = np.array([
    #    [3, 0, 0, 0, 0],
    #    [0, 2, 0, 0, 0],
    #    [0, 0, 2, 0, 0],
    #    [0, 0, 0, 2, 0],
    #    [0, 0, 0, 0, 1]
    #])
    
    (mathbf_I, mathbf_D) = naive_mathbf_I_and_D(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    mathbf_L = mathbf_D - mathbf_W
    
    mathbf_M = (1+beta) * mathbf_I + beta * mathbf_D + mathbf_L
    
    return mathbf_M

def naive_bar_s_M_and_M_prim(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    bar_s_M = (alpha+delta)/(alpha*n)*e_top_s    
    bar_s_M_prim = (1-alpha-delta)/((1-alpha)*n)*e_top_s
    return (bar_s_M, bar_s_M_prim)

def naive_e_M_and_M_prim_top_M_inv(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    (n_1, n_2, k) = naive_dims(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    mathbf_M = naive_mathbf_M(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    (bar_s_M, bar_s_M_prim) = naive_bar_s_M_and_M_prim(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    z_M = (1+gamma)*bar_s_M
    print(z_M)
    if trunc == False:
        assert z_M <= 1
    
    mathbf_M_inv = np.linalg.inv(mathbf_M)
    
    mathbf_e_M_top = np.block([
        [np.ones((1,n_1)), np.zeros((1,n_2))]
    ])
    mathbf_e_M_prim_top = np.block([
        [np.zeros((1,n_1)), np.ones((1,n_2))]
    ])
    
    e_M_top_M_inv = mathbf_e_M_top @ mathbf_M_inv
    
    #Taking the scalars only from the first index in each block.
    #U_1 = e_top_M_inv[0, 0]
    #print("U_1:")
    #print(U_1)
    #U_2 = e_top_M_inv[0, k]
    #V_1 = e_top_M_inv[0, n_1]
    #V_2 = e_top_M_inv[0, n_1+k]
    
    e_M_prim_top_M_inv = mathbf_e_M_prim_top @ mathbf_M_inv
    
    return (e_M_top_M_inv, e_M_prim_top_M_inv)

def naive_sum_of_expressed_equilibrium_case2(alpha, beta, gamma, delta, n, k, e_top_s, trunc = False):
    (n_1, n_2, k) = naive_dims(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    (bar_s_M, bar_s_M_prim) = naive_bar_s_M_and_M_prim(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    (e_M_top_M_inv, e_M_prim_top_M_inv) = naive_e_M_and_M_prim_top_M_inv(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
    
    
    
    #X_1 = e_top_M_prim_inv[0, 0]
    #print("X_1:")
    #print(X_1)
    #X_2 = e_top_M_prim_inv[0, k]
    #Y_1 = e_top_M_prim_inv[0, n_1]
    #Y_2 = e_top_M_prim_inv[0, n_1+k]
    
    zeta = np.block([
        (1+gamma)*(alpha+delta)/(n_1) * np.ones((k)),
        (1+gamma)*(alpha+delta)/(n_1) * np.ones((n_1-k)),
        (1-gamma)*(1-alpha-delta)/(n_2) * np.ones((k)),
        (1-gamma)*(1-alpha-delta)/(n_2) * np.ones((n_2-k))
    ]) * e_top_s
    print(zeta)
    
    # "Made up" s vector with avarage in each entry

    mathbf_s = np.block([
        bar_s_M * np.ones((n_1)),
        bar_s_M_prim * np.ones((n_2))
    ])
    print("mathbf_s:")
    print(mathbf_s)

    (mathbf_I, mathbf_D) = naive_mathbf_I_and_D(alpha, beta, gamma, delta, n, k, e_top_s, trunc)

    e_M_top_tilde_z_equi      = e_M_top_M_inv      @ (mathbf_s + beta*(mathbf_I + mathbf_D) @ zeta)

    e_M_prim_top_tilde_z_equi = e_M_prim_top_M_inv @ (mathbf_s + beta*(mathbf_I + mathbf_D) @ zeta)
    
    print("e_M_top_tilde_z_equi:")
    print(e_M_top_tilde_z_equi)
    
    sum = e_M_top_tilde_z_equi + e_M_prim_top_tilde_z_equi
    print("sum:")
    print(sum)
    
    return (e_M_top_tilde_z_equi, e_M_prim_top_tilde_z_equi)
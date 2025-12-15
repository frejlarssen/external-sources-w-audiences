import numpy as np
import src.numerical.num_caseI as numI

def thm_dims(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    if verify: #Round of if comparing to numerical results
        n_1 = int(np.rint(alpha * n))
        n_2 = int(np.rint(n-n_1))
        k = int(np.rint(k))
    else:
        n_1 = alpha*n
        n_2 = (1-alpha)*n
    return (n_1, n_2, k)

def var_a(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    return (1+beta)*n + 1

def sum_of_equilibrium_nontrunc(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    a = var_a(params, verify=False)
    
    
    #if verify:
    #    mathbf_s = e_top_s / n * np.ones(n) # Assume uniform distribution.
    #    #Thm 10
    #    thm_z_equi = 
    
    denom = beta*n+1
    
    M_nom = 1+beta*n*((1+gamma)*(alpha+delta)+(1-gamma)*(1-alpha-delta))
    e_M_top_z_equi = 1/a * ((alpha+delta)*(1+beta*n*(1+gamma))+\
                            M_nom / denom * alpha * n) * e_top_s

    M_prim_nom = (1+beta*n*((1+gamma)*(alpha+delta)+(1-gamma)*(1-alpha-delta)))
    e_M_prim_top_z_equi = 1/a * ((1-alpha-delta)*(1+beta*n*(1-gamma)) +\
                                  M_prim_nom/denom * (1-alpha)*n)*e_top_s
    return (e_M_top_z_equi, e_M_prim_top_z_equi)

def sum_of_equilibrium_trunc(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    a = var_a(params, verify=False)
    
    b_1 = (1+beta*n*(1-gamma)*(1-alpha-delta)) / ((beta * n + 1)*a) * alpha * n
    
    b_2 = (1-alpha)*(n+beta*(1-gamma)*n**2) / (beta*n+1)
    
    e_M_top_z_equi = alpha*beta*n**2/a + alpha**2*beta*n**3/((beta*n+1)*a) + ((alpha+delta)/a+b_1)*e_top_s
    
    e_M_prim_top_z_equi = alpha*(1-alpha)*beta*n**3/(beta*n+1) + ((1-alpha-delta)/a * (1+beta*(1-gamma)*n+b_2)) * e_top_s
    
    return (e_M_top_z_equi, e_M_prim_top_z_equi)

def sum_of_equilibrium(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    (n_1, n_2, k) = thm_dims(params, verify)

    # Check which formula to use
    z_M_test = (1+gamma)*(  alpha+delta)/n_1 * e_top_s
    
    print("thm.z_M_test: ", z_M_test)
    
    if verify:
        if z_M_test < 1:
            print("------VERIFYING NON-TRUNKACTED FORMULA------")
        else:
            print("--------VERIFYING TRUNKACTED FORMULA--------")

    return np.where(z_M_test < 1, sum_of_equilibrium_nontrunc(params, verify),
                                  sum_of_equilibrium_trunc(params, verify))

def avg_of_equilibrium_M(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    (alpha, beta, gamma, delta, n, k, e_top_s) = params
    
    print("(thm.n_1, thm.n_2, thm.n, thm.k): ", (n_1, n_2, n, k))
    
    e_M_top_z_equi = sum_of_equilibrium(params, verify)[0]
    
    if verify:
        (num_e_M_top_z_equi, num_e_M_prim_top_z_equi) = numI.sum_of_equilibrium(params)
        print("num_e_M_top_z_equi:")
        print(num_e_M_top_z_equi)
        print("e_M_top_z_equi:")
        print(e_M_top_z_equi)
        assert np.allclose(e_M_top_z_equi, num_e_M_top_z_equi), "e_M_top_z_equi incorrect"
    
    return e_M_top_z_equi / n_1

def avg_of_equilibrium_M_prim(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    e_M_prim_top_z_equi = sum_of_equilibrium(params, verify)[1]

    if verify:
        (num_e_M_top_z_equi, num_e_M_prim_top_z_equi) = numI.sum_of_equilibrium(params)
        print("num_e_M_prim_top_z_equi:")
        print(num_e_M_prim_top_z_equi)
        print("e_M_prim_top_z_equi:")
        print(e_M_prim_top_z_equi)
        assert np.allclose(e_M_prim_top_z_equi, num_e_M_prim_top_z_equi), "e_M_prim_top_z_equi incorrect"

    return e_M_prim_top_z_equi / n_2
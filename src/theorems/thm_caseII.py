import numpy as np
import src.numerical.num_caseII as numII

def thm_dims(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s, trunc) = params
    print("alpha: ", alpha)
    if verify:
        n_1 = int(np.rint(alpha * n))
        n_2 = int(n-n_1)
        k = int(k)
    else:
        n_1 = alpha*n
        n_2 = (1-alpha)*n
    return (n_1, n_2, k)

def thm_M_inv_params(params, verify=False):
    (alpha, beta, gamma, delta, n, k, e_top_s, trunc) = params
    (n_1, n_2, k) = thm_dims(params, verify)

    a_1 = 2.0 + beta + (beta + 1) * n_1
    d_1 = 2.0 + beta + (beta + 1) * n_2
    a_2 = 1.0+(1+beta) * n_1
    d_2 = 1.0+(1+beta) * n_2

    A = a_1*a_2-a_2*k-a_1*(n_1-k)
    D = d_1*d_2-d_2*k-d_1*(n_2-k)
    
    if verify:
        mathbf_M = numII.naive_mathbf_M(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
        mathcal_A = mathbf_M[0:n_1, 0:n_1]
        mathcal_B = mathbf_M[0:n_1, n_1:n]
        mathcal_C = mathbf_M[n_1:n, 0:n_1]
        mathcal_D = mathbf_M[n_1:n, n_1:n]
        
        thm_mathcal_A = np.block([
            [a_1 * np.eye(k), np.zeros((k, n_1-k))],
            [np.zeros((n_1-k, k)), a_2 * np.eye(n_1-k)]
        ]) - np.ones((n_1, n_1))
        
        print("thm_mathcal_A")
        print(thm_mathcal_A)
        assert np.allclose(thm_mathcal_A, mathcal_A)
        
        thm_mathcal_B = -np.block([
            [np.eye(k), np.zeros((k, n_2-k))],
            [np.zeros((n_1-k,k)), np.zeros((n_1-k,n_2-k))]
        ])
        
        print("thm_mathcal_B")
        print(thm_mathcal_B)
        assert np.allclose(thm_mathcal_B, mathcal_B)
        
        thm_mathcal_A_inv = 1/(a_1*a_2)*np.block([
            [a_2*np.eye(k), np.zeros((k, n_1-k))],
            [np.zeros((n_1-k, k)), a_1*np.eye(n_1-k)]
        ]) + 1/(a_1*a_2*A) * np.block([
            [a_2**2* np.ones((k,k)), a_1*a_2*np.ones((k, n_1-k))],
            [a_1*a_2*np.ones((n_1-k,k)), a_1**2*np.ones((n_1-k, n_1-k))]
        ])
        
        
        mathcal_A_inv = np.linalg.inv(mathcal_A)

        
        print("mathcal_A_inv")
        print(mathcal_A_inv)
        print("thm_mathcal_A_inv")
        print(thm_mathcal_A_inv)
        assert np.allclose(thm_mathcal_A_inv, mathcal_A_inv)

    p_a_nom = (1+a_2/(a_1*A))*a_1**2
    p_d_nom = (1+d_2/(d_1*D))*d_1**2
    p_a_denom = (a_1*d_1-1)*(a_1*d_1-1-a_1*(1+a_2/(a_1*A))*k)
    p_d_denom = (d_1*a_1-1)*(d_1*a_1-1-d_1*(1+d_2/(d_1*D))*k)
    p_a = p_a_nom / p_a_denom
    p_d = p_d_nom / p_d_denom
    
    s_a = 1/(d_2*(d_2-(n_2-k)))
    s_d = 1/(a_2*(a_2-(n_1-k)))

    P_a_nom = (a_1 * k +(a_1*d_1-1)*(p_a*k**2+1))*(a_1*d_1-1)
    P_d_nom = (d_1 * k +(d_1*a_1-1)*(p_d*k**2+1))*(d_1*a_1-1)
    P_a_denom = (a_1*d_1-1)*((a_1*d_1-1)*d_2**2-d_2*(a_1*k +(a_1*d_1-1)*(p_a*k**2+1))*(n_2-k))
    P_d_denom = (d_1*a_1-1)*((d_1*a_1-1)*a_2**2-a_2*(d_1*k +(d_1*a_1-1)*(p_d*k**2+1))*(n_1-k))
    P_a = P_a_nom / P_a_denom
    P_d = P_d_nom / P_d_denom

    S_a_nom = a_1**2*((1+a_2/(a_1*A))*d_2+(n_2-k)+(n_2-k)**2*s_a*d_2)*d_2
    S_d_nom = d_1**2*((1+d_2/(d_1*D))*a_2+(n_1-k)+(n_1-k)**2*s_d*a_2)*a_2
    S_a_denom = (a_1*d_1-1)*d_2*(d_2*(a_1*d_1-1) -a_1*k*((1+a_2/(a_1*A))*d_2+(n_2-k)+(n_2-k)**2*s_a*d_2))
    S_d_denom = (d_1*a_1-1)*a_2*(a_2*(d_1*a_1-1) -d_1*k*((1+d_2/(d_1*D))*a_2+(n_1-k)+(n_1-k)**2*s_d*a_2))
    S_a = S_a_nom / S_a_denom
    S_d = S_d_nom / S_d_denom

    Q_a=a_1/((a_1*d_1-1)*d_2)+(a_1*P_a*(n_2-k))/(a_1*d_1-1)+(p_a*k)/(d_2)+k*(n_2-k)*p_a*P_a
    Q_d=d_1/((d_1*a_1-1)*a_2)+(d_1*P_d*(n_1-k))/(d_1*a_1-1)+(p_d*k)/(a_2)+k*(n_1-k)*p_d*P_d

    v_1 = a_1/(a_1*d_1-1)
    v_2 = v_1+k*S_a
    
    if verify:
        mathbf_M = numII.naive_mathbf_M(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
        mathbf_M_inv = np.linalg.inv(mathbf_M)
        
        mathbf_U = mathbf_M_inv[0:n_1, 0:n_1]
        mathbf_V = mathbf_M_inv[0:n_1, n_1:n]
        mathbf_X = mathbf_M_inv[n_1:n, 0:n_1]
        mathbf_Y = mathbf_M_inv[n_1:n, n_1:n]
        
        
        thm_mathbf_U = np.block([
            [(d_1/(d_1*a_1-1))*np.eye(k) + S_d*np.ones((k,k)), Q_d * np.ones((k, n_1-k))],
            [Q_d*np.ones((n_1-k, k)), 1/a_2 * np.eye(n_1-k) + P_d*np.ones((n_1-k, n_1-k))]
        ])
    
        print("thm_mathbf_U: ")
        print(thm_mathbf_U)
        assert np.allclose(thm_mathbf_U, mathbf_U), "thm_mathbf_U incorrect"
        
        thm_M_schur_A_inv = np.block([
            [a_1/(a_1*d_1-1)*np.eye(k) + S_a*np.ones((k,k)), Q_a*np.ones((k,n_2-k))],
            [Q_a*np.ones((n_2-k,k)), 1/d_2*np.eye(n_2-k) + P_a*np.ones((n_2-k,n_2-k))]
        ])
        
        prod_right = mathcal_B @ mathbf_Y
        
        thm_prod_right = thm_mathcal_B @ thm_M_schur_A_inv
        
        thm_mathbf_B_mul_M_schur_A_inv = -np.block([
            [a_1/(a_1*d_1-1)*np.eye(k)+S_a*np.ones((k,k)), Q_a*np.ones((k,n_2-k))],
            [np.zeros((n_1-k,k)), np.zeros((n_1-k,n_2-k))]
        ])
        
        print("prod_right:")
        print(prod_right)
        
        print("thm_prod_right:")
        print(thm_prod_right)
        
        print("thm_mathbf_B_mul_M_schur_A_inv:")
        print(thm_mathbf_B_mul_M_schur_A_inv)
        
        assert np.allclose(thm_mathbf_B_mul_M_schur_A_inv, prod_right), "thm_mathbf_B_mul_M_schur_A_inv incorrect"
        
        thm_mathbf_V = 1/(a_1*A)*np.block([
            [v_1*A*np.eye(k)+(A*S_a+a_2*v_2)*np.ones((k,k)), (A+k*a_2)*Q_a*np.ones((k,n_2-k))],
            [a_1*v_2*np.ones((n_1-k,k)), k*a_1*Q_a*np.ones((n_1-k,n_2-k))]
        ])
        
        #old
        #thm_mathbf_V = 1/(a_1*a_2*A) * np.block([
        #    [a_2*A*v*np.eye(k) + (a_2**2*v+k*a_2**2*S_a+a_2*A*S_a)*np.ones((k,k)), (k*a_2**2*Q_a+a_2*A*Q_a)*np.ones((k,n_2-k))],
        #    [a_1*a_2*(v+k*S_a)*np.ones((n_1-k,k)), k*a_1*a_2*Q_a*np.ones((n_1-k,n_2-k))]
        #])

        thm_prod = -thm_mathcal_A_inv @ thm_mathcal_B @ thm_M_schur_A_inv

        print("thm_prod:")
        print(thm_prod)
        
        print("mathbf_V:")
        print(mathbf_V)
        
        print("thm_mathbf_V:")
        print(thm_mathbf_V)

        assert np.allclose(thm_mathbf_V, mathbf_V), "thm_mathbf_V incorrect"

        thm_mathbf_X = np.transpose(thm_mathbf_V)
        
        print("thm_mathbf_X:")
        print(thm_mathbf_X)
        assert np.allclose(thm_mathbf_X, mathbf_X), "thm_mathbf_X incorrect"
        
        thm_mathbf_Y = np.block([
            [a_1/(a_1*d_1-1)*np.eye(k)+S_a*np.ones((k,k)), Q_a*np.ones((k,n_2-k))],
            [Q_a*np.ones((n_2-k,k)), 1/d_2*np.eye(n_2-k)+P_a*np.ones((n_2-k,n_2-k))]
        ])

        print("thm_mathbf_Y:")
        print(thm_mathbf_Y)
        assert np.allclose(thm_mathbf_Y, mathbf_Y), "thm_mathbf_Y incorrect"
        
        thm_mathbf_M_inv = np.block([
            [thm_mathbf_U, thm_mathbf_V],
            [thm_mathbf_X, thm_mathbf_Y]
        ])
        
        print("mathbf_M_inv:")
        print(mathbf_M_inv)
        
        print("thm_mathbf_M_inv:")
        print(thm_mathbf_M_inv)
        
        assert np.allclose(thm_mathbf_M_inv, mathbf_M_inv), "thm_mathbf_M_inv incorrect"

    thm_U_1 = d_1/(d_1*a_1-1) + k*S_d+(n_1-k)*Q_d
    thm_U_2 = 1/a_2 +(n_1-k)*P_d +k*Q_d
    thm_V_1 = (v_1*A+k*(A*S_a+a_2*v_2)+(n_1-k)*a_1*v_2)/(a_1*A)
    thm_V_2 = (k*(A+k*a_2)*Q_a+(n_1-k)*k*a_1*Q_a)/(a_1*A)

    thm_X_1 = (v_1*A+k*(A*S_a+a_2*v_2)+(n_2-k)*(A+k*a_2)*Q_a)/(a_1*A)
    thm_X_2 = (k*a_1*v_2+(n_2-k)*k*a_1*Q_a)/(a_1*A)
    thm_Y_1 = a_1/(a_1*d_1-1) + k*S_a + (n_2-k)*Q_a
    thm_Y_2 = k*Q_a + 1/d_2+(n_2-k)*P_a
    
    if verify:
        (e_M_top_M_inv, e_M_prim_top_M_inv) = numII.naive_e_M_and_M_prim_top_M_inv(alpha, beta, gamma, delta, n, k, e_top_s, trunc)
        
        thm_e_M_top_M_inv = np.block(
            [thm_U_1*np.ones(k), thm_U_2*np.ones(n_1-k), thm_V_1*np.ones(k), thm_V_2*np.ones(n_2-k)]
        )
        
        print("e_M_top_M_inv:")
        print(e_M_top_M_inv)
        print("thm_e_M_top_M_inv:")
        print(thm_e_M_top_M_inv)     
        assert np.allclose(thm_e_M_top_M_inv, e_M_top_M_inv), "thm_e_M_top_M_inv incorrect"
        
        thm_e_M_prim_top_M_inv = np.block(
            [thm_X_1*np.ones(k), thm_X_2*np.ones(n_1-k), thm_Y_1*np.ones(k), thm_Y_2*np.ones(n_2-k)]
        )
        
        print("e_M_prim_top_M_inv:")
        print(e_M_prim_top_M_inv)
        print("thm_e_M_prim_top_M_inv:")
        print(thm_e_M_prim_top_M_inv)     
        assert np.allclose(thm_e_M_prim_top_M_inv, e_M_prim_top_M_inv), "thm_e_M_prim_top_M_inv incorrect"
    
    return (thm_U_1, thm_U_2, thm_V_1, thm_V_2, thm_X_1, thm_X_2, thm_Y_1, thm_Y_2)


#thm:sum-of-expressed-equilibrium-case2-nontrunc
def thm_sum_of_expressed_equilibrium_case2(params, verify=False, z_avg_all=False):
    (alpha, beta, gamma, delta, n, k, e_top_s, trunc) = params
    (n_1, n_2, k) = thm_dims(params, verify)
    (thm_U_1, thm_U_2, thm_V_1, thm_V_2, thm_X_1, thm_X_2, thm_Y_1, thm_Y_2) = thm_M_inv_params(params, verify=False)        

    # Opinion vectors
    if z_avg_all:
        thm_z_M =      (1+gamma)/n * e_top_s
        thm_z_M_prim = (1-gamma)/n * e_top_s
    else:
        thm_z_M =      (1+gamma)*(  alpha+delta)/n_1 * e_top_s
        thm_z_M_prim = (1-gamma)*(1-alpha-delta)/n_2 * e_top_s
    
    thm_bar_s_M =      (alpha+delta)/n_1 * e_top_s
    thm_bar_s_M_prim = (1-alpha-delta)/(n_2) * e_top_s

    #Total
    thm_e_M_top_tilde_z_equi      = thm_U_1*k      *(thm_bar_s_M      + beta*thm_z_M     *(1+n_1)) + \
                                    thm_U_2*(n_1-k)*(thm_bar_s_M      + beta*thm_z_M     *n_1    ) + \
                                    thm_V_1*k      *(thm_bar_s_M_prim + beta*thm_z_M_prim*(1+n_2)) + \
                                    thm_V_2*(n_2-k)*(thm_bar_s_M_prim + beta*thm_z_M_prim*n_2    )
    
    thm_e_M_prim_top_tilde_z_equi = thm_X_1*k      *(thm_bar_s_M      + beta*thm_z_M     *(1+n_1)) + \
                                    thm_X_2*(n_1-k)*(thm_bar_s_M      + beta*thm_z_M     *n_1    ) + \
                                    thm_Y_1*k      *(thm_bar_s_M_prim + beta*thm_z_M_prim*(1+n_2)) + \
                                    thm_Y_2*(n_2-k)*(thm_bar_s_M_prim + beta*thm_z_M_prim*n_2    )
    
    if verify:
        (e_M_top_tilde_z_equi, e_M_prim_top_tilde_z_equi) = numII.naive_sum_of_expressed_equilibrium_case2(alpha, beta, gamma, delta, n, k, e_top_s, trunc)

        print("e_M_top_tilde_z_equi:")
        print(e_M_top_tilde_z_equi)
        print("thm_e_M_top_tilde_z_equi:")
        print(thm_e_M_top_tilde_z_equi)
        assert np.isclose(thm_e_M_top_tilde_z_equi, e_M_top_tilde_z_equi), "thm_e_M_top_tilde_z_equi incorrect"

        print("e_M_prim_top_tilde_z_equi:")
        print(e_M_prim_top_tilde_z_equi)
        print("thm_e_M_prim_top_tilde_z_equi:")
        print(thm_e_M_prim_top_tilde_z_equi)
        assert np.isclose(thm_e_M_prim_top_tilde_z_equi, e_M_prim_top_tilde_z_equi), "thm_e_M_prim_top_tilde_z_equi incorrect"
        
        print("All asserts passed!")

    return (thm_e_M_top_tilde_z_equi, thm_e_M_prim_top_tilde_z_equi)

#thm_mathbf_M_inv = thm_sum_of_expressed_equilibrium_case2(alpha, beta, gamma, delta, k, n, mathbf_e_top_s, trunc=False, verify=True)


def thm_sum_of_expressed_equilibrium_case2_M(params, verify=False, z_avg_all=False):
    return thm_sum_of_expressed_equilibrium_case2(params, verify, z_avg_all)[0]
    
def thm_sum_of_expressed_equilibrium_case2_M_prim(params, verify=False, z_avg_all=False):
    return thm_sum_of_expressed_equilibrium_case2(params, verify, z_avg_all)[1]

def thm_sum_of_expressed_equilibrium_case2_tot(params, verify=False, z_avg_all=False):
    return thm_sum_of_expressed_equilibrium_case2(params, verify, z_avg_all)[0] +\
           thm_sum_of_expressed_equilibrium_case2(params, verify, z_avg_all)[1]


def thm_avg_of_expressed_equilibrium_case2_M(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_M(params, verify) / n_1

def thm_avg_of_expressed_equilibrium_case2_M_prim(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_M_prim(params, verify) / n_2

def thm_avg_of_expressed_equilibrium_case2_tot(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_tot(params, verify) / (n_1+n_2)


def out_avg_of_expressed_equilibrium_case2_M(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_M(params, verify, z_avg_all=True) / n_1

def out_avg_of_expressed_equilibrium_case2_M_prim(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_M_prim(params, verify, z_avg_all=True) / n_2

def out_avg_of_expressed_equilibrium_case2_tot(params, verify=False):
    (n_1, n_2, k) = thm_dims(params, verify)
    return thm_sum_of_expressed_equilibrium_case2_tot(params, verify, z_avg_all=True) / (n_1+n_2)
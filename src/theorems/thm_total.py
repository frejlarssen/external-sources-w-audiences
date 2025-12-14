import numpy as np
import src.numerical.num_total as num_tot

#thm:sum-equilibrium-expressed-opinions-non-trunk
def thm_sum_equilibrium_expressed_opinions_non_trunc(params):
    (alpha, beta, gamma, delta, d, n, e_top_s) = params
    return (1+(d+1)*beta*(1+gamma*(2*alpha+2*delta-1))) / (beta*(d+1)+1) * e_top_s

#thm:sum-equilibrium-expressed-opinions-trunk
def thm_sum_equilibrium_expressed_opinions_trunc(params):
    (alpha, beta, gamma, delta, d, n, e_top_s) = params
    return ((1+beta*(1+d)*(1-alpha-delta)*(1-gamma))*e_top_s + alpha*beta*(1+d)*n) / (1+beta*(1+d))

def thm_sum_equilibrium_expressed_opinions_general(params):
    (alpha, beta, gamma, delta, d, n, e_top_s) = params
    z_M = (1+gamma)*(alpha+delta) / (alpha*n) * e_top_s
    return np.where(z_M < 1, thm_sum_equilibrium_expressed_opinions_non_trunc(params),
                             thm_sum_equilibrium_expressed_opinions_trunc(params))

def thm_avg_equilibrium_expressed_opinions_general(params):
    (alpha, beta, gamma, delta, d, n, e_top_s) = params
    return thm_sum_equilibrium_expressed_opinions_general(params) / n
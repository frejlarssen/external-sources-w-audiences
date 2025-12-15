import numpy as np

#Applying Theorem 1 (thm:equilibrium)
def equilibrium(mathbf_I, mathbf_D, mathbf_L, mathbf_s, beta, zeta):
    #Matrix we want to take the inverse of
    mathbf_M = (1+beta)*mathbf_I + beta*mathbf_D + mathbf_L
    #print("numgen.mathbf_M:")
    #print(mathbf_M)
    second_factor = mathbf_s + beta*(mathbf_I+mathbf_D) @ zeta
    
    #print("numgen.second_factor:")
    #print(second_factor)
    
    z_equi = np.linalg.solve(mathbf_M, second_factor)
    return z_equi
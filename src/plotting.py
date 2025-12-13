import numpy as np
import matplotlib.pyplot as plt
import re

def sanitize_label(label):
    label = re.sub(r'mathbf', '', label)
    label = re.sub(r'tilde', '', label)
    return re.sub(r'[^a-zA-Z0-9_-]', '', label)

# Comparing up to four functions agains some x_values
def cmp_funcs(alpha, beta, gamma, delta, k, n, e_top_s, trunc,
                       x_values, x_label, y_label,
                       T_func1, T_func1_label, T_func2=None, T_func2_label=None, 
                       T_func3=None, T_func3_label=None, T_func4=None, T_func4_label=None,
                       T_func5=None, T_func5_label=None,
                       T_func6=None, T_func6_label=None,
                       yscale='linear'):

    y_values1 = T_func1(alpha, beta, gamma, delta, k, n, e_top_s, trunc)

    plt.plot(x_values, y_values1, label=T_func1_label, linestyle='--')
    if (T_func2):
        y_values2 = T_func2(alpha, beta, gamma, delta, k, n, e_top_s, trunc)
        plt.plot(x_values, y_values2, label=T_func2_label, linestyle='-.')
    if (T_func3):
        y_values3 = T_func3(alpha, beta, gamma, delta, k, n, e_top_s, trunc)
        plt.plot(x_values, y_values3, label=T_func3_label, linestyle=':')
    if (T_func4):
        y_values4 = T_func4(alpha, beta, gamma, delta, k, n, e_top_s, trunc)
        plt.plot(x_values, y_values4, label=T_func4_label, linestyle='-')
    if (T_func5):
        y_values5 = T_func5(alpha, beta, gamma, delta, k, n, e_top_s, trunc)
        plt.plot(x_values, y_values5, label=T_func5_label, linestyle='-')
    if (T_func6):
        y_values6 = T_func6(alpha, beta, gamma, delta, k, n, e_top_s, trunc)
        plt.plot(x_values, y_values6, label=T_func6_label, linestyle='-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.yscale(yscale)
    plt.grid(True)
    plt.legend()
    #plt.tight_layout()

    #filename = f"{sanitize_label(T_func1_label)}"
    #if T_func2:
    #    filename += f"_and_{sanitize_label(T_func2_label)}"
    #if T_func3:
    #    filename += f"_and_{sanitize_label(T_func3_label)}"
    #if T_func4:
    #    filename += f"_and_{sanitize_label(T_func4_label)}"
    #if T_func5:
    #    filename += f"_and_{sanitize_label(T_func5_label)}"
    #if T_func6:
    #    filename += f"_and_{sanitize_label(T_func6_label)}"
    #filename += f"-vs-{sanitize_label(x_label)}"
#
    #if not isinstance(alpha, np.ndarray):
    #    filename += f"-alpha{alpha}"
    #if not isinstance(beta, np.ndarray):
    #    filename += f"-beta{beta}"
    #if not isinstance(gamma, np.ndarray):
    #    filename += f"-gamma{gamma}"
    #if trunc:
    #    filename += f"-trunc"
    #else:
    #    filename += f"-nontrunc"
    #filename += f"-n{n}-e_top_s{e_top_s}.pdf"
#
    #plt.savefig("../continuous_plots/" + filename)
    plt.show()
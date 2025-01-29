import numpy as np

z = np.array([3.0, 2.0, 1.0, 0.1])
s = np.exp(z)/np.sum(np.exp(z)) 

def softmax_derivative(s):
    n = len(s)
    j_m = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                j_m[i, j] = s[i]*(1-s[j])
            else:
                j_m[i, j] = s[i]*(-s[j])
    return j_m

jacobian = softmax_derivative(s)
print(jacobian)


"""s = softmax(z). i is the position in the array of whatever number that softmax spits out. 
    j is the position in the array of what we feed into softmax."""
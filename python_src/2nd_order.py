import pandas as pd 
import numpy as np 



if __name__ == '__main__':
    N = np.array([])
    S = np.array([1.5, 1.3, 1.2])
    L = np.array([2.8, 3.5, 4.8])

    # Prework
    a = np.dot(S,S)
    b = np.dot(S,L)
    c = np.dot(L,L)

    # Step 1
    coeffs = [4*c*(a*c-b**2), -4*(a*c-b**2), a+2*b+c-4*a*c, 2*(a-b), a-1]
    roots = np.roots(coeffs)
    y = roots[roots > 0]
    print(y)

    # Step 2
    x = (-2*c*y**2 + y + 1) / (2*b*y + 1)
    i = np.argwhere(x > 0)[0][0]

    # Good x,y
    N = x[i]*S + y[i]*L
    print('(x,y): \t',(x[i],y[i]))
    print('N: \t', N)
    print('||N||: \t', np.linalg.norm(N))
from scipy.linalg import svd, toeplitz, tril
import numpy as np

def trefethen(cn: list, m: int, n: int):
    assert m + n < len(cn), 'm + n + 1 must be less than the length of cn'
    
    if all([ci == 0 for ci in cn]):
        return [0], [1]

    if n == 0:
        return cn[:m+1], [1]

    c = np.zeros((m+n+1, n+1))

    for row in range(n+1):
        # print(f'{row}: {cn[row::-1]}')
        c[row, :row+1] = cn[row::-1]
    for row in range(n+1, m+n+1):
        # print(f'{row}: {cn[row:row-n-1:-1]}')
        c[row, :] = cn[row:row-n-1:-1]
        # print(row-(n+1))

    c_upper = c[:m+1, :]
    c_tilde = c[m+1:, :]

    U,S,V = svd(c_tilde)
    print(U)
    print(S)
    print(V)

    p = np.count_nonzero(S)

    if p < n:
        return trefethen(cn, m-(n-p), p)
    else:
        b = V[:, -1]
        a = np.dot(c_upper, b)
        print(a, b)



    # print(c_tilde)


    # return p,q





if __name__ == '__main__':
    m = 5
    n = 3
    cs = [0,1,2,3,4,5,6,7,8,9,10,11,12]
    print(trefethen(cs, m, n))
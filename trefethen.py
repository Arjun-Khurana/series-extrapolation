from scipy.linalg import svd, toeplitz, tril, norm
from scipy.interpolate import pade
import numpy as np
from matplotlib import pyplot as plt

def trefethen(cn: list, m: int, n: int = -1, tol: float = 1e-14):
    if n == -1: 
        n = len(cn) - m - 1
    assert m + n + 1 <= len(cn), 'm + n + 1 must be less than or equal to the length of cn'
    
    if not any(cn):
        return [0], [1]

    if n == 0:
        return cn[:m+1], [1]

    c = np.zeros((m+n+1, n+1))
    norm_cn = norm(cn)

    for row in range(n+1):
        # print(f'{row}: {cn[row::-1]}')
        c[row, :row+1] = cn[row::-1]
    for row in range(n+1, m+n+1):
        # print(f'{row}: {cn[row:row-n-1:-1]}')
        c[row, :] = cn[row:row-n-1:-1]
        # print(row-(n+1))

    c_upper = c[:m+1, :]
    c_tilde = c[m+1:, :]

    _,S,V = svd(c_tilde)
    # print(U)
    # print(S)
    # print(V)

    p = len(S) - np.count_nonzero([s if np.abs(s) <= tol*norm_cn else 0 for s in S])
    print(p)

    if p < n:
        return trefethen(cn, m-(n-p), p)
    else:
        b = V[:, -1]
        a = np.dot(c_upper, b)
        print(a)
        print(b)
        for i,bi in enumerate(b):
            if np.abs(bi) <= tol:
                b[i] = 0
                a[i] = 0
            else:
                break

        a_lam = [i for i, ai in enumerate(np.flip(a)) if np.abs(ai) <= tol][0]+1
        b_lam = [i for i, bi in enumerate(np.flip(b)) if np.abs(bi) <= tol][0]+1
        # # print(a_lam)
        # # print(b_lam)
        a = a[:a_lam]
        b = b[:b_lam]
        # print(a)
        # print(b)
        a = np.divide(a, b[0])
        b = np.divide(b, b[0])
        print(a)
        print(b)
        
        return np.poly1d(np.flip(a)), np.poly1d(np.flip(b))

def sin_data():
    T = 1
    N = 10
    x = np.linspace(0, T, N)
    f = 1
    # y = np.sin(2*np.pi * f * x) + np.sin(2*np.pi * 2*f * x)
    y = np.sin(2*np.pi * f * x)

    min_freq, max_freq = -4, 4
    freq_range = max_freq - min_freq
    freqs = np.linspace(min_freq, max_freq, freq_range * N)

    step = 1
    dt = x[step] - x[0]

    return {
        'y': y,
        'dt': dt,
        'step': step,
        'freqs': freqs
    }

def ring_data():
    with np.load(f'ring-data_res=20/ring-data_t=200.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']
        freqs = data['freqs']

    t = np.arange(0, len(ez)*dt, dt)[:len(ez)]

    start = 1
    stop = len(ez)
    step = 4

    samp = list(zip(t, ez))[start:stop:step]
    t_samp, ez_samp = zip(*samp)

    return {
        'y': ez_samp,
        'dt': dt,
        'step': step,
        'freqs': freqs
    }

if __name__ == '__main__':
    data = sin_data()

    y = data['y']
    dt = data['dt']
    step = data['step']
    freqs = data['freqs']

    N = len(y)
    

    m = int(N/2)
    a_tref,b_tref = trefethen(y, m)
    # m = int(N/2)
    a,b = pade(y, m)

    P = lambda w: np.sum([pi * np.exp(1j * w * step*dt * n) for n,pi in enumerate((a))])
    Q = lambda w: np.sum([qi * np.exp(1j * w * step*dt * n) for n,qi in enumerate((b))])
    P_tref = lambda w: np.sum([pi * np.exp(1j * w * step*dt * n) for n,pi in enumerate(((a_tref)))])
    Q_tref = lambda w: np.sum([qi * np.exp(1j * w * step*dt * n) for n,qi in enumerate(((b_tref)))])

    freq_domain = []
    freq_domain_tref = []

    for freq in freqs:
        freq_domain.append(P(2*np.pi*freq)/Q(2*np.pi*freq))
        freq_domain_tref.append(P_tref(2*np.pi*freq)/Q_tref(2*np.pi*freq))

    plt.semilogy(freqs, np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), label='pade')
    plt.semilogy(freqs, np.abs(freq_domain_tref)**2 / max(np.abs(freq_domain_tref)**2), label='trefethen')
    plt.legend()
    plt.show()

    # tmp = np.array([[ai*(xi**i) for xi in x] for i, ai in enumerate(a)])
    # tmp_b = np.array([[bi*(xi**i) for xi in x] for i, bi in enumerate(b)])
    # tmp = np.sum(tmp, axis=0)
    # tmp_b = np.sum(tmp_b, axis=0)
    # np.divide(tmp, tmp_b)

    # plt.plot(x, np.divide(tmp, tmp_b))
    # plt.plot(x, y)
    # plt.show()

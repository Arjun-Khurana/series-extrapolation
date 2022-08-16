# from scipy.interpolate import pade
from sandbox import pade
import numpy as np
from matplotlib import pyplot as plt
import argparse

def delta(n):
    return 1 if n == 0 else 0

def analytic_pade(p,q,n):
    numerator = 0

    for i,pn in enumerate(np.flip(p)):
        numerator += pn * delta(n - i)

    denominator = q[-1]

    for i,qn in enumerate(np.flip(q[:-1])):
        denominator += qn * delta(n - i)

    return numerator/denominator

def ring_pade(args) -> None:
    with np.load(f'ring-data_t={args.time}.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']
        freqs = data['freqs']

    with np.load(f'cavity-data_N=0.npz') as data:
        dft2 = data['dft']

    print(len(ez))

    t = np.arange(0, len(ez)*dt, dt)
    print(len(t))
    plt.plot(t, ez)
    plt.savefig(f'time-domain_t={args.time}.png')
    plt.close()

    start = 0
    stop = len(ez)
    step = 1

    samp = list(zip(t, ez))[start:stop:step]
    t_samp, ez_samp = zip(*samp)
    # plt.scatter(t_samp, ez_samp)
    # plt.show()

    p,q = pade(ez_samp, int(len(ez_samp)/2))
    print('pade done')

    pn = p.coef
    qn = q.coef

    P = lambda w: np.sum([pi * np.exp(1j * w * dt * n) for n,pi in enumerate(np.flip(pn))])
    Q = lambda w: np.sum([qi * np.exp(1j * w * dt * n) for n,qi in enumerate(np.flip(qn))])

    freq_domain = []

    for freq in freqs:
        freq_domain.append(P(2*np.pi*freq)/Q(2*np.pi*freq))

    plt.plot(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label='meep')
    plt.plot(freqs, np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), label='pade')
    plt.legend()
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$ (a.u.)')
    plt.savefig(f'extrapolation_t={t[-1]}.png')

    np.savez(f'pade-data_t={args.time}',
        p=p,
        q=q,
        pade_approx=freq_domain
    )

def cavity_pade(args):
    with np.load(f'cavity-data_t={args.time}.npz') as data:
        ey = data['ey']
        dft = data['dft']
        fcen, df, dt = data['domain']
        freqs = data['freqs']

    with np.load(f'cavity-data_N=0.npz') as data:
        dft2 = data['dft']

    print(len(ey))

    t = np.arange(0, len(ey)*dt, dt)
    print(len(t))
    plt.plot(t, ey)
    plt.savefig(f'time-domain_t={args.time}.png')
    plt.close()

    start = 0
    stop = len(ey)
    step = 1

    samp = list(zip(t, ey))[start:stop:step]
    t_samp, ey_samp = zip(*samp)
    # plt.scatter(t_samp, ez_samp)
    # plt.show()

    p,q = pade(ey_samp, int(len(ey_samp)/2))
    print('pade done')

    pn = p.coef
    qn = q.coef

    P = lambda w: np.sum([pi * np.exp(1j * w * dt * n) for n,pi in enumerate(np.flip(pn))])
    Q = lambda w: np.sum([qi * np.exp(1j * w * dt * n) for n,qi in enumerate(np.flip(qn))])

    freq_domain = []

    for freq in freqs:
        freq_domain.append(P(2*np.pi*freq)/Q(2*np.pi*freq))

    meep_dft = np.abs(dft/dft2)**2
    pade_dft = np.abs(freq_domain/dft2)**2

    plt.plot(freqs, meep_dft / max(meep_dft), label='meep')
    plt.plot(freqs, pade_dft / max(pade_dft), label='pade')
    plt.legend()
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$ (a.u.)')
    plt.savefig(f'extrapolation_t={t[-1]}.png')

    np.savez(f'pade-data_t={args.time}',
        p=p,
        q=q,
        pade_approx=freq_domain
    )

# def cavity_baseline(args):
#     for i in [3, 5, 7]:
#         with np.load(f'cavity-data_N={i}.npz') as data:
#             ey = data['ey']
#             dft = data['dft']
#             fcen, df, dt = data['domain']
#             freqs = data['freqs']
#             t = data['t']
        
#         with np.load(f'cavity-data_N=0.npz') as data:
#             dft2 = data['dft']

#         plt.plot(freqs, np.abs(dft/dft2)**2, label='meep')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=200, help='time')
    parser.add_argument('--numholes', '-N', type=int, default=3, help='number of holes')
    args = parser.parse_args()
    cavity_pade(args)
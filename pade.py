from scipy.interpolate import pade
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

def main(args) -> None:
    with np.load(f'ring-data_t={args.time}.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']
        freqs = data['freqs']

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

    pn = p.coef
    qn = q.coef

    P = lambda w: np.sum([pi * np.exp(1j * w * dt * n) for n,pi in enumerate(np.flip(pn))])
    Q = lambda w: np.sum([qi * np.exp(1j * w * dt * n) for n,qi in enumerate(np.flip(qn))])

    freq_domain = []

    for freq in freqs:
        freq_domain.append(P(2*np.pi*freq)/Q(2*np.pi*freq))

    plt.semilogy(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label='meep')
    plt.semilogy(freqs, np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), label='pade')
    plt.legend()
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$ (a.u.)')
    plt.savefig(f'extrapolation_t={t[-1]}.png')

    np.savez(f'pade-data_t={args.time}',
        p=p,
        q=q,
        pade_approx=freq_domain
    )
    # plt.plot(freqs, np.abs(dft)**2 / np.abs(freq_domain)**2)
    # plt.show()

    # plt.plot(freqs, np.angle(dft))
    # plt.plot(freqs, np.angle(freq_domain))
    # plt.show()

    # time_domain = np.fft.ifft(freq_domain)
    # plt.plot(time_domain)
    # plt.show()
    # print(qn[-1])

    # pade_an = []

    # for n,tn in enumerate(t_samp):
    #     pade_an.append(analytic_pade(pn, qn, n))

    # plt.plot(t_samp, pade_an)
    # plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=200, help='time')
    args = parser.parse_args()
    main(args)
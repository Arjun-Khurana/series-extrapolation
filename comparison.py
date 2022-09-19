import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import trapz
import argparse

# x = np.linspace(-10, 10, 1000)
# y = np.sin(2*np.pi*x)

# fw = np.fft.fft(y)
# w = np.fft.fftfreq(len(x), x[1]-x[0])

# # y2 = np.fft.ifft(fw)
# # plt.plot(x,y2)
# # plt.show()
# plt.plot(w,fw)
# plt.show()

# plt.plot(x,y)
# plt.show()


def main(args):
    ts = [args.padetime, args.dtfttime, args.groundtime]
    f = args.file

    for t in ts:
        with np.load(f'{f}_t={t}.npz') as data:
            dft = data['dft']
            freqs = data['freqs']

        # periods = (5*t)//200

        plt.semilogy(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label=f'$|DTFT|^2$, {t}')

    plt.legend()
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$ (a.u.)')
    plt.savefig(f'dft-extension.png')
    plt.close()


    ring_data = np.load(f'{f}_t={args.groundtime}.npz')
    pade_data = np.load(f'pade-data_t={args.padetime}.npz')
    freqs = ring_data['freqs']
    pade_approx = pade_data['pade_approx']
    dft = ring_data['dft']

    plt.semilogy(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label=f'meep, {args.groundtime}')
    plt.semilogy(freqs, np.abs(pade_approx)**2 / max(np.abs(pade_approx)**2), label=f'pade, {args.padetime}')
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$ (a.u.)')
    plt.legend()
    plt.savefig(f'pade-winning.png')
    plt.close()

    ring_data = np.load(f'{f}_t={args.padetime}.npz')
    pade_data = np.load(f'pade-data_t={args.padetime}.npz')
    freqs = ring_data['freqs']
    pade_approx = pade_data['pade_approx']
    dft = ring_data['dft']
    plt.semilogy(freqs, np.abs(dft)**2, label='dtft')
    plt.semilogy(freqs, np.abs(pade_approx)**2, label='pade')
    plt.xlabel('frequency (meep units)')
    plt.ylabel('$|DTFT|^2$')
    plt.legend()
    plt.savefig('unnormalized.png')

# with np.load(f'ring-data_t={40000}.0.npz') as data:
#     ez = data['ez']
#     dft = data['dft']
#     fcen, df, dt = data['domain']
#     freqs = data['freqs']

# print(len(ez))

# t = np.arange(0, len(ez)*dt, dt)
# print(len(t))
# plt.plot(t,ez)
# # plt.semilogy(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label='normalized dtft: 16 periods')
# plt.xlabel('time (meep units)')
# plt.ylabel('$E_z$')
# plt.savefig(f'time-domain_t={40000}.png')
# plt.close()
# quit()

# with np.load(f'ring-data_t={4000}.0.npz') as data:
#     ez = data['ez']
#     dft = data['dft']
#     fcen, df, dt = data['domain']
#     freqs = data['freqs']

# print(len(ez))

# dft_4k = dft
# t = np.arange(0, len(ez)*dt, dt)
# print(len(t))
# plt.semilogy(freqs, np.abs(dft)**2 / max(np.abs(dft)**2), label='normalized dtft: 100 periods')
# plt.legend()
# plt.xlabel('frequency (meep units)')
# plt.ylabel('$|DTFT|^2$ (a.u.)')
# plt.savefig(f'dft-extension.png')
# plt.close()

# ts = [75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# # ts = [75]

# int_pade = []
# int_dft = []

# for t in ts:
#     ring_data = np.load(f'ring-data_t={t}.0.npz')
#     pade_data = np.load(f'pade-data_t={t}.0.npz')
#     freqs = ring_data['freqs']
#     pade_approx = pade_data['pade_approx']
#     dft = ring_data['dft']
#     int_pade.append(trapz(np.abs(pade_approx)**2 / max(np.abs(pade_approx)**2), freqs))
#     int_dft.append(trapz(np.abs(dft)**2 / max(np.abs(dft)**2), freqs))


# # int_dft.append(trapz(np.abs(dft)**2 / max(np.abs(dft)**2), freqs))

# plt.scatter(ts, int_pade)
# plt.scatter(ts, int_dft)
# plt.scatter([*ts, 4000], int_dft)
# plt.show()




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--padetime', '-pt', type=int, default=200, help='time for pade')
    parser.add_argument('--dtfttime', '-dt', type=int, default=4000, help='time for dtft')
    parser.add_argument('--groundtime', '-gt', type=int, default=10000, help='time for groundtruth')
    parser.add_argument('--file', '-f', type=str, help='file prefix')
    args = parser.parse_args()
    main(args)
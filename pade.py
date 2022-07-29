from scipy.interpolate import approximate_taylor_polynomial, pade
import numpy as np
from matplotlib import pyplot as plt

def main() -> None:
    with np.load('ring-data.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']

    # fw = 0

    plt.plot(dft)
    plt.show()
    fw = np.fft.fft(ez)
    plt.plot(fw)
    plt.xlim([0, 100])
    plt.show()

    # t = np.arange(0, len(ez)*dt, dt)
    # # print(t)
    # plt.plot(t, ez)
    # plt.savefig('time-domain.png')
    # # print(t[-1])

    # fw = np.sum(dt/np.sqrt(2*np.pi) * (ez * np.exp(1j * 2*np.pi*(fcen) * t)))

    # print(f'abs(dft): {np.abs(dft)**2}')
    # print(f'abs(fw): {np.abs(fw)**2}')

    # print(f'ang(dft): {np.angle(dft)}')
    # print(f'ang(fw): {np.angle(fw)}')



if __name__ == '__main__':
    main()
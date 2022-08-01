from scipy.interpolate import approximate_taylor_polynomial, pade
import numpy as np
from matplotlib import pyplot as plt

def main() -> None:
    with np.load('ring-data.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']

    # plt.plot(dft)
    # plt.savefig('dtft.png')
    # plt.close()
    # fw = np.fft.fftshift(np.fft.fft(ez))
    # plt.plot(np.abs(fw)**2)
    # plt.plot(np.angle(fw)**2)
    # plt.xlim([0, 100])
    # plt.show()
    # quit()

    t = np.arange(0, len(ez)*dt, dt)
    # print(t)
    plt.plot(t, ez)
    plt.savefig('time-domain.png')
    # print(t[-1])

    fw = 0
    for n,fn in enumerate(ez):
        fw += dt/np.sqrt(2*np.pi) * (fn * np.exp(1j * 2*np.pi*(fcen) * n * dt))
    
    print(f'abs(dft): {np.abs(dft)**2}')
    print(f'abs(fw): {np.abs(fw)**2}')

    print(f'ang(dft): {np.angle(dft)}')
    print(f'ang(fw): {np.angle(fw)}')



if __name__ == '__main__':
    main()
import numpy as np
from matplotlib import pyplot as plt

def main() -> None:
    with np.load('ring-data.npz') as data:
        ez = data['ez']
        dft = data['dft']
        fcen, df, dt = data['domain']
        freqs = data['freqs']

    t = np.arange(0, len(ez)*dt, dt)
    # print(t)
    plt.plot(t, ez)
    plt.savefig('time-domain.png')
    print(t[-1])

    fws = []
    for freq in freqs:
        fw = 0
        for n,fn in enumerate(ez):
            fw += dt/np.sqrt(2*np.pi) * (fn * np.exp(1j * 2*np.pi*(freq) * n * dt))
        fws.append(fw/4)


    print(f'abs(dft): {np.abs(dft)**2}')
    plt.plot(np.abs(dft), label='meep')
    plt.plot(np.abs(fws), label='manual')
    plt.xlabel('frequency')
    plt.ylabel('abs(dtft)')
    plt.legend()
    plt.savefig('abs_dtft.png')
    plt.close()
    plt.plot(np.angle(dft), label='meep')
    plt.plot(np.angle(fws), label='manual')
    plt.xlabel('frequency')
    plt.ylabel('angle(dtft)')
    plt.legend()
    plt.savefig('angle_dtft.png')

    plt.plot(np.abs(fws / dft))
    plt.show()
    plt.plot(np.angle(fws / dft))
    plt.show()
    print(f'abs(fw): {np.abs(fws)**2}')

    print(f'ang(dft): {np.angle(dft)}')
    print(f'ang(fw): {np.angle(fws)}')

if __name__ == '__main__':
    main()
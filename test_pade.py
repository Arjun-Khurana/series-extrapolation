import meep as mp
import numpy as np
import argparse
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, peak_prominences

def main(time) -> None:
    n = 3.4                 # index of waveguide
    w = 1                   # width of waveguide
    r = 1                   # inner radius of ring
    pad = 4                 # padding between waveguide and edge of PML
    dpml = 2                # thickness of PML
    sxy = 2*(r+w+pad+dpml)  # cell size

    c1 = mp.Cylinder(radius=r+w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)
    b1 = mp.Block(size=mp.Vector3(sxy, w, 0), material=mp.Medium(index=n))

    fcen = 0.15              # pulse center frequency
    df = 0.1                 # pulse frequency width
    dtft_df = 0.05
    # freqs = [fcen]
    src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))

    sim = mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy),
        geometry=[c1, c2],
        sources=[src],
        resolution=20,                    
        boundary_layers=[mp.PML(dpml)]
    )


    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(r + w/2, 0, 0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    # sim.plot2D()
    # plt.show()

    p = mp.Pade(
        c = mp.Ez,
        pt = mp.Vector3(r+0.1),
        # m_frac = 0.5,
        # n_frac = 0.45,
        sample_rate=4
    )

    h = mp.Harminv(mp.Ez, mp.Vector3(r + 0.1), fcen, df)

    sim.run(
        p,
        mp.after_sources(h),
        until_after_sources=time-100
    )
    # print(len(p.data))
    # print(p.m)
    # print(p.n)

    freq_domain = []

    for freq in dtft_freqs:
        freq_domain.append(p.freq_response(freq))

    idx = find_peaks(np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), prominence=1e-3)[0]
    print(peak_prominences(np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), idx))
    # print(idx)
    print(dtft_freqs[idx])
    # print((np.abs(freq_domain)**2)[idx])

    dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]

    plt.semilogy(dtft_freqs, np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2), label='pade')
    plt.scatter(dtft_freqs[idx], (np.abs(freq_domain)**2 / max(np.abs(freq_domain)**2))[idx])
    # plt.semilogy(dtft_freqs, np.abs(dft_data)**2 / max(np.abs(dft_data)**2), '--', label='meep')
    plt.legend()
    plt.savefig(f'time={time}.png')
    plt.close()


if __name__ == '__main__':
    main(300)
    # for time in np.linspace(50, 300, 6):
    #     main(200)

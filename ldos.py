import meep as mp
import numpy as np
from matplotlib import pyplot as plt
import argparse

def main(args) -> None:
    w = 20
    dpml = 2
    resolution = 20

    fcen = 2
    df = 3.5

    sim = mp.Simulation(
        cell_size = (w,1),
        boundary_layers=[mp.PML(dpml, direction=mp.X)],
        sources=[
            mp.Source(
                src=mp.GaussianSource(fcen,fwidth=df*2),
                component=mp.Ez,
                center=mp.Vector3()
            )
        ],
        symmetries=[mp.Mirror(mp.X), mp.Mirror(mp.Y)],
        resolution=resolution
    )

    ldos = mp.Ldos(fcen, df, 300)
    sim.run(
        mp.dft_ldos(ldos=ldos),
        until_after_sources=args.time
    )

    plt.plot(mp.get_ldos_freqs(ldos), np.abs(sim.ldos_data))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=200, help='time')
    args = parser.parse_args()
    main(args)
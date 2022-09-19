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
    dtft_df = 1.75
    dt = 0

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

    ez_data = []

    def get_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ez, mp.Vector3(x=0))
        ez_data.append(x)

    def save_data(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
        if mp.am_really_master():
            np.savez(f'ldos-data_t={int(time*dt)}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(x=0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    sim.plot2D()
    plt.savefig('geometry.png')

    sim.run(
        mp.at_every(dt, get_field),
        # mp.at_every(args.dpt, save_data),
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(x=0), fcen, df)),
        until=args.padetime
    )

    sim.run(
        mp.at_every(dt, get_field),
        # mp.at_every(args.ddt, save_data),
        # mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+w/2), fcen, df)),
        until=args.dtfttime-args.padetime
    )

    sim.run(
        mp.at_every(args.dgt, save_data),
        until=args.groundtime-args.dtfttime
    )

    # save_data(sim)

    # ldos = mp.Ldos(fcen, dtft_df, 1000)

    # sim.run(
    #     mp.at_every(dt, save_field),
    #     until=args.time
    # )

    # dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
    # dt = sim.fields.dt

    # if mp.am_really_master():
    #     np.savez(f'ldos-data_t={args.time}.npz', 
    #     ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    # plt.semilogy(dtft_freqs, np.abs(dft_data)**2 / max(np.abs(dft_data)**2))
    # plt.savefig(f'dtft_t={args.time}.png')
    # plt.close()

    # plt.semilogy(mp.get_ldos_freqs(ldos), np.abs(sim.ldos_data))
    # plt.savefig(f'ldos_t={args.time}.png')
    # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', '-r', type=float, default=30, help='resolution')
    parser.add_argument('--padetime', '-pt', type=float, default=200, help='time for pade')
    parser.add_argument('-dpt', type=float, default=50, help='differential time for pade')
    parser.add_argument('--dtfttime', '-dt', type=float, default=2000, help='time for dtft')
    parser.add_argument('-ddt', type=float, default=500, help='differential time for dtft')
    parser.add_argument('--groundtime', '-gt', type=float, default=10000, help='time for ground truth')
    parser.add_argument('-dgt', type=float, default=500, help='differential time for ground truth')
    args = parser.parse_args()
    main(args)
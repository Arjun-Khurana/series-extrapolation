import meep as mp
import numpy as np
import argparse
from matplotlib import pyplot as plt

def main(args) -> None:
    n = 3.4                 # index of waveguide
    w = 1                   # width of waveguide
    r = 1                   # inner radius of ring
    pad = 4                 # padding between waveguide and edge of PML
    dpml = 2                # thickness of PML
    sxy = 2*(r+w+pad+dpml)  # cell size

    c1 = mp.Cylinder(radius=r+w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)

    fcen = 0.175              # pulse center frequency
    df = 0.1                 # pulse frequency width
    dtft_df = 0.05
    chron = 0.5
    dt = 0
    freqs = np.linspace(fcen - df, fcen + df, 100)
    # freqs = [fcen]
    src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))

    sim = mp.Simulation(
        cell_size=mp.Vector3(sxy, sxy),
        geometry=[c1, c2],
        sources=[src],
        resolution=args.resolution,                    
        boundary_layers=[mp.PML(dpml)]
    )


    ez_data = []

    def get_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ez, mp.Vector3(r + w/2, 0, 0))
        ez_data.append(x)

    def save_data(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
        if mp.am_really_master():
            np.savez(f'ring-data_t={int(time*dt)}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(r + w/2, 0, 0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    sim.plot2D()
    plt.savefig('geometry.png')

    sim.run(
        mp.at_every(dt, get_field),
        mp.at_every(args.dpt, save_data),
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+w/2), fcen, df)),
        until=args.padetime
    )

    sim.run(
        mp.at_every(dt, get_field),
        mp.at_every(args.ddt, save_data),
        # mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+w/2), fcen, df)),
        until=args.dtfttime-args.padetime
    )

    sim.run(
        mp.at_every(args.dgt, save_data),
        until=args.groundtime-args.dtfttime
    )

    save_data(sim)

    # dt = sim.fields.dt
    # dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
    # if mp.am_really_master():
    #     np.savez(f'ring-data_t={args.time}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    # sim.run(
    #     mp.at_beginning(mp.output_epsilon),
    #     mp.to_appended("ez", mp.at_every(1, mp.output_efield_z)),
    #     until_after_sources=800
    # )

    # sim.run(
    #     mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+w/2), fcen, df)),
    #     until_after_sources=4000
    # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', '-r', type=float, default=30, help='resolution')
    parser.add_argument('--padetime', '-pt', type=float, default=200, help='time for pade')
    parser.add_argument('-dpt', type=float, default=50, help='differential time for pade')
    parser.add_argument('--dtfttime', '-dt', type=float, default=4000, help='time for dtft')
    parser.add_argument('-ddt', type=float, default=500, help='differential time for dtft')
    parser.add_argument('--groundtime', '-gt', type=float, default=10000, help='time for ground truth')
    parser.add_argument('-dgt', type=float, default=500, help='differential time for ground truth')
    args = parser.parse_args()
    main(args)
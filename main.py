import meep as mp
import numpy as np
import argparse

def main(args) -> None:
    n = 3.4                 # index of waveguide
    w = 1                   # width of waveguide
    r = 1                   # inner radius of ring
    pad = 4                 # padding between waveguide and edge of PML
    dpml = 2                # thickness of PML
    sxy = 2*(r+w+pad+dpml)  # cell size

    c1 = mp.Cylinder(radius=r+w, material=mp.Medium(index=n))
    c2 = mp.Cylinder(radius=r)

    fcen = 0.15              # pulse center frequency
    df = 0.1                 # pulse frequency width
    dtft_df = 0.02
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

    def save_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ez, mp.Vector3(r + w/2, 0, 0))
        ez_data.append(x)

    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(r + w/2, 0, 0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    sim.run(
        mp.at_every(dt, save_field),
        until=args.time
    )
    dt = sim.fields.dt
    dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
    if mp.am_really_master():
        np.savez(f'ring-data_t={args.time}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    # sim.run(
    #     mp.at_beginning(mp.output_epsilon),
    #     mp.to_appended("ez", mp.at_every(1, mp.output_efield_z)),
    #     until_after_sources=800
    # )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', '-r', type=float, default=30, help='resolution')
    parser.add_argument('--time', '-t', type=float, default=200, help='time')
    args = parser.parse_args()
    main(args)
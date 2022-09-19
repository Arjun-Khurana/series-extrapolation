import meep as mp
import numpy as np
import argparse
from matplotlib import pyplot as plt

def normalization(args):
    nx = 0
    pad = 4
    dpml = 1
    fcen = 0.5
    df = 0.6
    dtft_df = 0.3
    sx = nx + 2*(pad+dpml)
    dt = 0

    sim = mp.Simulation(
        cell_size = mp.Vector3(sx, 1),
        geometry=mp.geometric_object_duplicates(
            mp.Vector3(1,0),
            0,
            nx-1,
            mp.Cylinder(
                0.2,
                center=mp.Vector3(-0.5*sx + dpml + pad + 0.5, 0),
                material=mp.Medium(epsilon=12)
            )
        ),
        boundary_layers=[mp.PML(dpml, direction=mp.X)],
        k_point=mp.Vector3(0,0),
        sources=[
            mp.Source(
                mp.GaussianSource(fcen,fwidth=df),
                mp.Ez,
                center=(-sx*0.5 + dpml, 0),
                size=(0,1)
            )
        ],
        symmetries=[mp.Mirror(mp.Y,phase=1)],
        resolution=args.resolution
    )

    fluxregion = mp.FluxRegion(
        center=mp.Vector3(0.5*sx-dpml),
        size=mp.Vector3(0,1)
    )

    trans = sim.add_flux(fcen, dtft_df, nfreq:=1000, fluxregion)

    ez_data = []

    def get_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ez, mp.Vector3(0.5*sx - dpml, 0, 0))
        ez_data.append(x)

    def get_flux(sim: mp.Simulation = None):
        x = sim.flux_in_box(
            mp.X,
            center=mp.Vector3(0.5*sx-dpml),
            size=mp.Vector3(0,1)
        )
        ez_data.append(x)

    def save_data(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
        if mp.am_really_master():
            np.savez(f'rods-data_n=0.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    def save_flux(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = np.array(mp.get_fluxes(trans))
        if mp.am_really_master():
            np.savez(f'rods-data_n=0.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(0.5*sx-dpml, 0, 0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    # sim.plot2D()
    # plt.savefig('geometry.png')

    sim.run(
        mp.at_every(dt, get_field),
        until_after_sources=mp.stop_when_fields_decayed(
            10, mp.Ez, mp.Vector3(0.5*sx - dpml), 1e-3
        )
    )

    # return sim, np.array(mp.get_flux_freqs(trans)), np.array(mp.get_fluxes(trans))

    save_data(sim)

def main(args) -> None:
    nx = args.nx
    pad = 4
    dpml = 1
    fcen = 0.5
    df = 0.6
    dtft_df = 0.3
    sx = nx + 2*(pad+dpml)
    dt = 0

    sim = mp.Simulation(
        cell_size = mp.Vector3(sx, 1),
        geometry=mp.geometric_object_duplicates(
            mp.Vector3(1,0),
            0,
            nx-1,
            mp.Cylinder(
                0.2,
                center=mp.Vector3(-0.5*sx + dpml + pad + 0.5, 0),
                material=mp.Medium(epsilon=12)
            )
        ),
        boundary_layers=[mp.PML(dpml, direction=mp.X)],
        k_point=mp.Vector3(0,0),
        sources=[
            mp.Source(
                mp.GaussianSource(fcen,fwidth=df),
                mp.Ez,
                center=(-sx*0.5 + dpml, 0),
                size=(0,1)
            )
        ],
        symmetries=[mp.Mirror(mp.Y,phase=1)],
        resolution=args.resolution
    )

    ez_data = []

    fluxregion = mp.FluxRegion(
        center=mp.Vector3(0.5*sx-dpml),
        size=mp.Vector3(0,1)
    )

    trans = sim.add_flux(fcen, dtft_df, nfreq:=1000, fluxregion)


    def get_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ez, mp.Vector3(0.5*sx - dpml, 0, 0))
        ez_data.append(x)

    def get_flux(sim: mp.Simulation = None):
        x = sim.flux_in_box(
            mp.X,
            center=mp.Vector3(0.5*sx-dpml),
            size=mp.Vector3(0,1)
        )
        ez_data.append(x)

    def save_data(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = [sim.get_dft_array(dft_fields, mp.Ez, i) for i in range(len(dtft_freqs))]
        if mp.am_really_master():
            np.savez(f'rods-data_t={int(time*dt)}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    def save_flux(sim: mp.Simulation = None):
        dt = sim.fields.dt
        time = sim.fields.t
        dft_data = np.array(mp.get_fluxes(trans))
        if mp.am_really_master():
            np.savez(f'rods-data_t={int(time*dt)}.npz', ez=ez_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    dft_fields = sim.add_dft_fields(
        [mp.Ez],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(0.5*sx-dpml, 0, 0),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    sim.plot2D()
    plt.savefig('geometry.png')

    sim.run(
        mp.at_every(dt, get_field),
        mp.at_every(args.dpt, save_data),
        mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(0.5*sx - dpml), fcen, df)),
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
    parser.add_argument('--resolution', '-r', type=float, default=20, help='resolution')
    parser.add_argument('--padetime', '-pt', type=float, default=100, help='time for pade')
    parser.add_argument('-dpt', type=float, default=10, help='differential time for pade')
    parser.add_argument('--dtfttime', '-dt', type=float, default=500, help='time for dtft')
    parser.add_argument('-ddt', type=float, default=100, help='differential time for dtft')
    parser.add_argument('--groundtime', '-gt', type=float, default=1000, help='time for ground truth')
    parser.add_argument('-dgt', type=float, default=100, help='differential time for ground truth')
    parser.add_argument('-nx', type=int, default=0, help='numrods')
    args = parser.parse_args()
    normalization(args)
    main(args)
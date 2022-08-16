import meep as mp
import numpy as np
import argparse
from matplotlib import pyplot as plt

def main(args) -> None:
    resolution = 30

    eps = 13  # dielectric constant of waveguide
    w = 1.2  # width of waveguide
    r = 0.36  # radius of holes
    d = 1.4  # defect spacing (ordinary spacing = 1)
    N = args.numholes  # number of holes on either side of defect

    sy = 6
    pad = 2
    dpml = 1

    sx = 2 * (pad + dpml + N) + d - 1

    dt = 0

    cell = mp.Vector3(sx, sy, 0)

    geometry = [
        mp.Block(
            size=mp.Vector3(mp.inf, w, mp.inf),
            material = mp.Medium(epsilon=eps)
        )
    ]

    for i in range(N):
        geometry.append(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)))
        geometry.append(mp.Cylinder(r, center=mp.Vector3(-(d / 2 + i))))

    fcen = 0.25
    df = 0.2
    dtft_df = 0.1

    source = mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ey,
        center=mp.Vector3(-0.5*sx + dpml + .5),
        size=mp.Vector3(0,w)
    )

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=[source],
        symmetries=[mp.Mirror(mp.Y, phase=-1)],
        boundary_layers=[mp.PML(dpml)],
        resolution=resolution
    )

    # sim.plot2D()
    # plt.show()
    # quit()

    dft_fields = sim.add_dft_fields(
        [mp.Ey],
        dtft_freqs:= np.linspace(fcen - dtft_df, fcen + dtft_df, 1000),
        center=mp.Vector3(0.5*sx-dpml-0.5),
        size=mp.Vector3(0,0,0),
        decimation_factor=1
    )

    ey_data = []

    def save_field(sim: mp.Simulation = None):
        x = sim.get_field_point(mp.Ey, mp.Vector3(0.5*sx-dpml-0.5))
        ey_data.append(x)

    # sim.run(
    #     mp.at_beginning(mp.output_epsilon),
    #     mp.to_appended("ey", mp.at_every(1, mp.output_efield_y)),
    #     until_after_sources=800
    # )

    sim.run(
        mp.at_every(dt, save_field),
        until=args.time
    )

    # sim.run(
    #     mp.at_every(dt, save_field),
    #     until_after_sources=mp.stop_when_fields_decayed(
    #         50, mp.Ey, mp.Vector3(0.5 * sx - dpml - 0.5), 1e-3
    #     )
    # )

    # dft_data = [sim.get_dft_array(dft_fields, mp.Ey, i) for i in range(len(dtft_freqs))]
    # if mp.am_really_master():
    #     np.savez(f'cavity-data_normal.npz', ey=ey_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs)

    dt = sim.fields.dt
    t = sim.fields.t
    dft_data = [sim.get_dft_array(dft_fields, mp.Ey, i) for i in range(len(dtft_freqs))]
    if mp.am_really_master():
        np.savez(f'cavity-data_t={args.time}.npz', ey=ey_data, dft=dft_data, domain=[fcen, df, dt], freqs=dtft_freqs, t=t)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', type=float, default=200, help='time')
    parser.add_argument('--numholes', '-N', type=int, default=3, help='number of holes')
    args = parser.parse_args()
    main(args)
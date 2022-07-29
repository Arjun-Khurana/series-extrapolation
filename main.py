import meep as mp

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
src = mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Ez, mp.Vector3(r+0.1))

sim = mp.Simulation(
    cell_size=mp.Vector3(sxy, sxy),
    geometry=[c1, c2],
    sources=[src],
    resolution=10,                    
    boundary_layers=[mp.PML(dpml)]
)

sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(1, mp.output_efield_z)),
    mp.after_sources(mp.Harminv(mp.Ez, mp.Vector3(r+0.1), fcen, df)),
    until_after_sources=300
)
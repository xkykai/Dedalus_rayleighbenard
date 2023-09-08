"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
This script demonstrates solving a 2D Cartesian initial value problem. It can
be ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_snapshots.py` script can be used to
produce plots from the saved data. It should take about 5 cpu-minutes to run.

The problem is non-dimensionalized using the box height and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""
#%%
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import argparse
#%%
parser = argparse.ArgumentParser()

parser.add_argument('--Ra', required=True, type=float)
parser.add_argument('--Ta', required=True, type=float)
parser.add_argument('--Nz', required=True, type=int)

parser.add_argument('--Pr', default=1, type=float)
parser.add_argument('--timestep', default=2e-6, type=float)
parser.add_argument('--save_dt', default=0.1, type=float)
parser.add_argument('--stop_time', default=20, type=float)
parser.add_argument('--aspect_ratio', default=0.25, type=float)
 
# Read arguments from command line
args = parser.parse_args()
print(args.Ra)

# Parameters
Lz = 1
Lx = Lz / args.aspect_ratio

Nz = args.Nz
Nx = int(Nz / args.aspect_ratio)

Rayleigh = args.Ra
Taylor = args.Ta
Prandtl = args.Pr

dealias = 3/2
save_dt = args.save_dt
stop_sim_time = args.stop_time
timestepper = d3.RK222
timestep = args.timestep
dtype = np.float64

print("end of variables")
# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
b = dist.Field(name='b', bases=(xbasis,zbasis))
u = dist.Field(name='u', bases=(xbasis,zbasis))
v = dist.Field(name='v', bases=(xbasis,zbasis))
w = dist.Field(name='w', bases=(xbasis,zbasis))

tau_p = dist.Field(name='tau_p')
tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
tau_u1 = dist.Field(name='tau_u1', bases=xbasis)
tau_u2 = dist.Field(name='tau_u2', bases=xbasis)
tau_v1 = dist.Field(name='tau_v1', bases=xbasis)
tau_v2 = dist.Field(name='tau_v2', bases=xbasis)
tau_w1 = dist.Field(name='tau_w1', bases=xbasis)
tau_w2 = dist.Field(name='tau_w2', bases=xbasis)

#%%
# # Substitutions
nu = 1
kappa = nu / Prandtl
S = Rayleigh * nu * kappa / Lz**4
f = (Taylor * nu**2 / Lz**4)**0.5

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)

grad_p = d3.grad(p)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_v = d3.grad(v) + ez*lift(tau_v1) # First-order reduction
grad_w = d3.grad(w) + ez*lift(tau_w1) # First-order reduction
grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction

dpdx = d3.Differentiate(p, coords['x'])
dbdx = d3.Differentiate(b, coords['x'])
dudx = d3.Differentiate(u, coords['x'])
dvdx = d3.Differentiate(v, coords['x'])
dwdx = d3.Differentiate(w, coords['x'])

dpdz = d3.Differentiate(p, coords['z'])
dbdz = d3.Differentiate(b, coords['z'])
dudz = d3.Differentiate(u, coords['z'])
dvdz = d3.Differentiate(v, coords['z'])
dwdz = d3.Differentiate(w, coords['z'])

#%%
# Problem
# First-order form: "div(f)" becomes "trace(grad_f)"
# First-order form: "lap(f)" becomes "div(grad_f)"
problem = d3.IVP([p, b, u, v, w, tau_p, tau_b1, tau_b2, tau_u1, tau_u2, tau_v1, tau_v2, tau_w1, tau_w2], namespace=locals())
problem.add_equation("grad_u@ex + grad_w@ez + tau_p = 0")
problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u*dbdx - w*dbdz")
problem.add_equation("dt(u) - nu*div(grad_u) - f*v + grad_p@ex + lift(tau_u2) = - u*dudx - w*dudz")
problem.add_equation("dt(v) - nu*div(grad_v) + f*u + lift(tau_v2) = - u*dvdx - w*dvdz")
problem.add_equation("dt(w) - nu*div(grad_w) + grad_p@ez - b + lift(tau_w2) = - u*dwdx - w*dwdz")
problem.add_equation("integ(p) = 0") # Pressure gauge

# w bcs
problem.add_equation("w(z=0) = 0")
problem.add_equation("w(z=Lz) = 0")

# b bcs
problem.add_equation("b(z=0) = 0")
problem.add_equation("b(z=Lz) = -S")

# u, v bcs
problem.add_equation("dudz(z=0) = 0")
problem.add_equation("dudz(z=Lz) = 0")
problem.add_equation("dvdz(z=0) = 0")
problem.add_equation("dvdz(z=Lz) = 0")

# problem.add_equation("u(z=0) = 0")
# problem.add_equation("u(z=Lz) = 0")
# problem.add_equation("v(z=0) = 0")
# problem.add_equation("v(z=Lz) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
#%%
# Initial conditions
b.fill_random('g', seed=42, distribution='normal', scale=Rayleigh/100) # Random noise
b['g'] *= z * (Lz - z) # Damp noise at walls
b['g'] += -S * z # Add linear background

# Analysis
Ra_str = "{:e}".format(Rayleigh).replace(".", "pt")
Ta_str = "{:e}".format(Taylor).replace(".", "pt")

snapshots = solver.evaluator.add_file_handler(f"Data/snapshots_Nz_{Nz}_Ra_{Ra_str}_Ta_{Ta_str}_test", sim_dt=save_dt, max_writes=50)
snapshots.add_task(b, name='buoyancy')
snapshots.add_task(u, name='u')
snapshots.add_task(v, name='v')
snapshots.add_task(w, name='w')
snapshots.add_task(np.sqrt(d3.Average(w**2)), name='w_rms')

# Flow properties
logging_cadence = 100
flow = d3.GlobalFlowProperty(solver, cadence=logging_cadence)
flow.add_property(np.sqrt(u**2 + v**2 + w**2)/nu, name='Re')
flow.add_property(np.abs(b), name='|b|')
flow.add_property(np.abs(u), name='|u|')
flow.add_property(np.abs(v), name='|v|')
flow.add_property(np.abs(w), name='|w|')
#%%
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = timestep
        solver.step(timestep)
        if (solver.iteration-1) % logging_cadence == 0:
            max_Re = flow.max('Re')
            max_b = flow.max('|b|')
            max_u = flow.max('|u|')
            max_v = flow.max('|v|')
            max_w = flow.max('|w|')
            logger.info('Iteration=%i, Time=%e, dt=%e, max: |Re|=%f, |b|=%f, |u|=%f, |v|=%f, |w|=%f' %(solver.iteration, solver.sim_time, timestep, max_Re, max_b, max_u, max_v, max_w))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
#%%
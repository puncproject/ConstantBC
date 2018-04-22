# Copyright (C) 2017, Sigvald Marholm and Diako Darian
#
# This file is part of ConstantBC.
#
# ConstantBC is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# ConstantBC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# ConstantBC.  If not, see <http://www.gnu.org/licenses/>.

from dolfin import *
from mshr import *
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from ConstantBC import *
from itertools import count
from numpy import log as ln

def solve_poisson(V, rho, bcs, objects=None, circuit=None):

    if objects==None:
        objects = []

    phi = TrialFunction(V)
    psi = TestFunction(V)

    a = dot(grad(phi), grad(psi)) * dx
    L = rho*psi * dx

    phi = Function(V)

    A = assemble(a)
    b = assemble(L)

    for bc in bcs:
        bc.apply(A, b)

    for o in objects:
        o.apply(A, b)

    if circuit != None:
        A, b = circuit.apply(A, b)

    # Direct solution is just as quick for small problems
    solve(A, phi.vector(), b)

#     solver = PETScKrylovSolver('gmres','hypre_amg')
#     solver.parameters['absolute_tolerance'] = 1e-14
#     solver.parameters['relative_tolerance'] = 1e-10 #e-12
#     solver.parameters['maximum_iterations'] = 100000
#     solver.parameters['monitor_convergence'] = False

#     solver.set_operator(A)
#     solver.solve(phi.vector(), b)

    for o in objects:
        o.correct_charge(phi)

    return phi

def get_bnd(mesh, boundaries):
    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    for i, boundary in enumerate(boundaries):
        boundary.mark(bnd, i+1)
    return bnd

orders      = [1,2]
resolutions = [1,2,3,4]#,5]#,6]#,7,8]

rho = Expression("100*x[1]", degree=1)

l = 0.2
ll = 1.0
EPS = DOLFIN_EPS

class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return (np.abs(x[0])>ll-EPS or np.abs(x[1])>ll-EPS) and on_bnd

class InnerBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return (np.abs(x[0])<l+EPS and np.abs(x[1])<l+EPS) and on_bnd

boundaries = [OuterBoundary(), InnerBoundary()]
n_bnd = len(boundaries)

vsources = [[-1,0,1]]   # Do not use charge constraint. Same as Dirichlet.
# vsources = []         # Use charge constraint.

domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
       - Circle(Point(0,0), l, 90)

#
# REFERENCE SOLUTION
#

mesh = generate_mesh(domain, 10)
for res in resolutions[:-1]:
    mesh = refine(mesh)
res += 1

order = 5
print("Order: {}, resolution: {} (reference)".format(order,res))

V_ref = FunctionSpace(mesh, 'CG', order)

bnd = get_bnd(mesh, boundaries)
bcs_ref = [DirichletBC(V_ref, Constant(i), bnd, i+1) for i in range(n_bnd)]
phi_ref = solve_poisson(V_ref, rho, bcs_ref)

n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)
Qs_ref = [assemble(dot(grad(phi_ref), n) * dss(i+1)) for i in range(1,n_bnd)]
Qtot = sum(Qs_ref)

#
# VERIFICATION SOLUTIONS
#

mesh = generate_mesh(domain, 10)

hmins = {}
errors = {}
for res in resolutions:

    bnd = get_bnd(mesh, boundaries)

    hmins[res] = mesh.hmin()
    errors[res] = {}
    for order in orders:

        print("Order: {}, resolution: {}".format(order,res))

        V = FunctionSpace(mesh, 'CG', order)
        bcs = [DirichletBC(V, Constant(0), bnd, 1)]
        objects = [ObjectBC(V, bnd, i+1) for i in range(1,n_bnd)]
        circuit = Circuit(V, bnd, objects, vsources)
        objects[0].charge = Qtot

        phi_sol = solve_poisson(V, rho, bcs, objects, circuit)

        error = errornorm(phi_ref, phi_sol, degree_rise=0, mesh=mesh)
        errors[res][order] = error

    mesh = refine(mesh)

for order in orders:
    x = np.array([hmins[k] for k in hmins.keys()])
    y = np.array([errors[k][order] for k in hmins.keys()])
    p = plt.loglog(x, y, '-o', label="$\\mathrm{{CG}}_{}$".format(order))

    color = p[0].get_color()
    plt.loglog(x, y[0]*(x/x[0])**(order+1), ':', color=color)

    r = np.zeros(y.shape)
    r[1:] = ln(y[1:]/y[:-1])/ln(x[1:]/x[:-1])

    print("order={}".format(order))
    for i in range(len(x)):
        print("h=%2.2E E=%2.2E r=%.2f" %(x[i], y[i], r[i]))

plt.grid()
plt.xlabel('$h_min$')
plt.ylabel('$L_2$ norm error')
plt.title('Convergence')
plt.legend(loc='lower right')
plt.show()

File('phi.pvd') << phi_sol

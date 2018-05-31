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
from ConstantBC import ConstantBoundary as ConstantBnd
from itertools import count
from numpy import log as ln

l = 0.2
ll = 1.0

class ConstantBoundary(SubDomain):

    def __init__(self, mesh, bnd):

        SubDomain.__init__(self)
        self.mesh = mesh

        # Pick a random vertex on the bnd (assuming this vertex is a node)
        facet_id    = bnd.where_equal(2)[0]
        facet       = list(facets(mesh))[facet_id]
        vertex_id   = facet.entities(0)[0]
        self.vertex = mesh.coordinates()[vertex_id]

    def inside(self, x, on_bnd):
        # Some FEniCS functions (not all) will pass 3D x even in 2D problems
        x = x[:self.mesh.geometry().dim()]
        return np.linalg.norm(x-self.vertex) < DOLFIN_EPS

    def map(self, x, y):
        on_sphere = np.linalg.norm(x)-l < DOLFIN_EPS
        if on_sphere and not self.inside(x, True):
            y[:] = self.vertex
        else:
            y[:] = x[:]

class InnerBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.all( np.abs(x) < l+DOLFIN_EPS ) and on_bnd

class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.any( np.abs(x) > ll-DOLFIN_EPS ) and on_bnd

def solve_poisson(mesh, rho, order, Q=0, method='DirichletBC'):

    assert method in ('DirichletBC','ConstantBC','ConstantBoundary')

    outer_bnd = OuterBoundary()
    inner_bnd = InnerBoundary()

    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, 1)
    inner_bnd.mark(bnd, 2)

    const_bnd = ConstantBoundary(mesh, bnd)

    n = FacetNormal(mesh)
    dss = Measure("ds", domain=mesh, subdomain_data=bnd)
    dsi = dss(2)
    S  = assemble(Constant(1.)*dsi)

    if method=='DirichletBC':
        V = FunctionSpace(mesh, 'CG', order)

        phi = TrialFunction(V)
        psi = TestFunction(V)

        ext_bc = DirichletBC(V, Constant(0), bnd, 1)
        int_bc = DirichletBC(V, Constant(1), bnd, 2)
        bcs = [ext_bc, int_bc]

        a = dot(grad(phi), grad(psi)) * dx
        L = rho*psi * dx

    elif method=='ConstantBoundary':
        V = FunctionSpace(mesh, 'CG', order, constrained_domain=const_bnd)

        phi = TrialFunction(V)
        psi = TestFunction(V)

        ext_bc = DirichletBC(V, Constant(0), bnd, 1)
        bcs = [ext_bc]

        a = dot(grad(phi), grad(psi)) * dx
        L = rho*psi * dx + psi*(Constant(Q)/S)*dsi

    else:
        V = FunctionSpace(mesh, 'CG', order)

        phi = TrialFunction(V)
        psi = TestFunction(V)

        ext_bc = DirichletBC(V, Constant(0), bnd, 1)
        int_bc = ConstantBC( V,              bnd, 2)
        bcs = [ext_bc, int_bc]

        a = dot(grad(phi), grad(psi)) * dx
        L = rho*psi * dx + psi*(Constant(Q)/S)*dsi


    phi_ = Function(V)

    # solve(a==L, phi_, bcs=bc)

    A = assemble(a)
    b = assemble(L)
    for bc in bcs:
        bc.apply(A, b)

    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = False

    solver.set_operator(A)
    solver.solve(phi_.vector(), b)

    Q_ = assemble(dot(grad(phi_), n) * dsi)

    return phi_, Q_, V

orders      = [1,2]
resolutions = [1,2,3]#,4,5]#,6]#,7,8]

rho = Expression("100*x[1]", degree=1)
# rho = Constant(0)

domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
       - Circle(Point(0,0), l, 90)

mesh = generate_mesh(domain, 10)
for res in resolutions:
    mesh = refine(mesh)
order = orders[-1]+1
res = resolutions[-1]+1

print("Order: {}, resolution: {} (reference)".format(order,res))
phi_ref, Q_ref, V_ref = solve_poisson(mesh, rho, order, method='DirichletBC')
print(Q_ref)

mesh = generate_mesh(domain, 10)
# plot(mesh); plt.show()

hmins = {}
errors = {}
for res in resolutions:

    hmins[res] = mesh.hmin()
    errors[res] = {}
    for order in orders:

        print("Order: {}, resolution: {}".format(order,res))
        phi_sol, Q_sol, V_sol = solve_poisson(mesh, rho, order, Q_ref, method='ConstantBoundary')
        print(Q_sol)

        File('phi_order{}_res{}.pvd'.format(order,res)) << phi_sol
        # File('error_order{}_res{}.pvd'.format(order,res)) << interpolate(phi_sol,V_ref)-phi_ref

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
plt.xlabel('$h_{min}$')
plt.ylabel('$L_2$ norm error')
plt.title('Convergence')
plt.legend(loc='lower right')
plt.show()

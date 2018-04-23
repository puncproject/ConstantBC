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

l = np.pi/2
ll = 3*l
a = ll**2/(ll**2-l**2)
c = 1/(l**2-ll**2)

rho = Expression("2*cos(x[0])*cos(x[1])", degree=4)
phi_e = Expression("cos(x[0])*cos(x[1])", degree=4)
bc_val = 0

EPS = DOLFIN_EPS
class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.any(np.abs(x)>=ll-EPS) and on_bnd

class InnerBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return np.all(np.abs(x)<=l+EPS) and on_bnd

def solve_poisson(mesh, order):

    V = FunctionSpace(mesh, 'CG', order)
    phi = TrialFunction(V)
    psi = TestFunction(V)

    a = dot(grad(phi), grad(psi)) * dx
    L = rho*psi * dx

    bc0 = DirichletBC(V, Constant(0), OuterBoundary())
    bc1 = DirichletBC(V, Constant(bc_val), InnerBoundary())
    phi = Function(V)

    # solve(a==L, phi, bcs=bc)

    A = assemble(a)
    b = assemble(L)
    bc0.apply(A, b)
    bc1.apply(A, b)

    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = False

    solver.set_operator(A)
    solver.solve(phi.vector(), b)

    phi_ref = interpolate(phi_e, V)

    return phi, phi_ref

orders              = [1,2]
resolutions         = [1,2,3,4,5]#,6]#,7,8]

domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
       - Rectangle(Point(-l,-l), Point(l,l))
# domain = Rectangle(Point(-ll,-ll), Point(ll,ll))

mesh = generate_mesh(domain, 10)
plot(mesh); plt.show()

hmins = {}
errors = {}
for res in resolutions:

    hmins[res] = mesh.hmin()
    errors[res] = {}
    for order in orders:

        print("Order: {}, resolution: {}".format(order,res))
        phi_sol, phi_ref = solve_poisson(mesh, order)

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

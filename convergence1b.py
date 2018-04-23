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

def solve_poisson(mesh, rho, order):

    V = FunctionSpace(mesh, 'CG', order)
    phi = TrialFunction(V)
    psi = TestFunction(V)

    a = dot(grad(phi), grad(psi)) * dx
    L = rho*psi * dx

    bc = DirichletBC(V, Constant(0), DomainBoundary())
    phi = Function(V)

    # solve(a==L, phi, bcs=bc)

    A = assemble(a)
    b = assemble(L)
    bc.apply(A, b)

    solver = PETScKrylovSolver('gmres','hypre_amg')
    solver.parameters['absolute_tolerance'] = 1e-14
    solver.parameters['relative_tolerance'] = 1e-10 #e-12
    solver.parameters['maximum_iterations'] = 100000
    solver.parameters['monitor_convergence'] = False

    solver.set_operator(A)
    solver.solve(phi.vector(), b)

    return phi

orders = [1,2,3,4,5,6,7,8]

rho = Expression("100*x[1]", degree=1)

l = 0.2
ll = 1.0

nsegs = [5,20,90]
for nseg in nsegs:

    domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
           - Circle(Point(0,0), l, nseg)

    mesh = generate_mesh(domain, 20)
    # plot(mesh); plt.show()

    print("Order: {} (reference)".format(orders[-1]))
    phi_ref = solve_poisson(mesh, rho, orders[-1])

    errors = []
    ndofs  = []
    for order in orders[:-1]:

        print("Order: {}".format(order))
        phi_sol = solve_poisson(mesh, rho, order)

        error = errornorm(phi_ref, phi_sol, degree_rise=0, mesh=mesh)
        errors.append(error)
        ndofs.append(len(phi_sol.vector().get_local()))

    p = plt.loglog(ndofs, errors, '-o', label="{} segments".format(nseg))
    color = p[0].get_color()

plt.grid()
plt.xlabel('Number of dofs')
plt.ylabel('$L_2$ norm error')
plt.title('Convergence')
plt.legend(loc='lower left')
plt.show()

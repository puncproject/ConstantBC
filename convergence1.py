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

orders              = [1,2]
resolutions         = [1,2,3,4,5]#,6]#,7,8]

rho = Expression("100*x[1]", degree=1)

l = 0.2
ll = 1.0
lh = 0.05
lo = 0.7
li = 0.1
EPS = DOLFIN_EPS

# domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
#        - Rectangle(Point(-l,-l), Point(l,l))
# domain = Rectangle(Point(-ll,-ll), Point(ll,ll))
# domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
#        - Rectangle(Point(-l,ll-2*l), Point(l,ll))
# domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
#        - Polygon([Point(-lo,ll), Point(-li,ll-lh), Point(li,ll-lh), Point(lo,ll)])
domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
       - Circle(Point(0,0), l, 80)

mesh = generate_mesh(domain, 10)
# mesh = Mesh("triangle.xml")
plot(mesh); plt.show()

hmins = {}
errors = {}
for res in resolutions:

    order = 5
    print("Order: {}, resolution: {} (reference)".format(order,res))

    W = FunctionSpace(mesh, 'CG', order)
    phi = TrialFunction(W)
    psi = TestFunction(W)
    a = dot(grad(phi), grad(psi)) * dx
    L = rho*psi * dx
    bc = DirichletBC(W, Constant(0), DomainBoundary())
    phi_ref = Function(W)
    solve(a==L, phi_ref, bcs=bc)

    hmins[res] = mesh.hmin()
    errors[res] = {}
    for order in orders:

        print("Order: {}, resolution: {}".format(order,res))

        V = FunctionSpace(mesh, 'CG', order)
        phi = TrialFunction(V)
        psi = TestFunction(V)
        a = dot(grad(phi), grad(psi)) * dx
        L = rho*psi * dx
        bc = DirichletBC(V, Constant(0), DomainBoundary())
        phi_sol = Function(V)
        solve(a==L, phi_sol, bcs=bc)

        error = errornorm(phi_ref, phi_sol, degree_rise=0, mesh=mesh)
        errors[res][order] = error

    mesh = refine(mesh)

for order in orders:
    x = np.array([hmins[k] for k in hmins.keys()])
    y = np.array([errors[k][order] for k in hmins.keys()])
    p = plt.loglog(x, y, '-o', label="$CG_{}$".format(order))

    color = p[0].get_color()
    plt.loglog(x, y[0]*(x/x[0])**(order+1), ':', color=color)

    r = np.zeros(y.shape)
    r[1:] = ln(y[1:]/y[:-1])/ln(x[1:]/x[:-1])

    print("order={}".format(order))
    for i in range(len(x)):
        print("h=%2.2E E=%2.2E r=%.2f" %(x[i], y[i], r[i]))

plt.grid()
plt.xlabel('hmin')
plt.ylabel('$L_2$ norm error')
plt.title('Convergence')
plt.legend(loc='lower right')
plt.show()

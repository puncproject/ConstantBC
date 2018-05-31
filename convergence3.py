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
resolutions         = [1,2,3,4,5]
res_start           = 2
monitor_convergence = True
allow_extrapolation = True

rho = Expression("100*x[1]", degree=1)

r = 1
d = 3*r
h = 3*r
l = d+3*r
EPS = DOLFIN_EPS

segments = 90
domain = Rectangle(Point(-l,-h), Point(l,h)) \
       - Circle(Point(-d,0), r, segments) \
       - Circle(Point( d,0), r, segments)

class OuterBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return (np.abs(x[0])>l-EPS or np.abs(x[1])>h-EPS) and on_bnd

class LeftObjectBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return norm(x-np.array([-d,0]))<r+EPS and on_bnd

class RightObjectBoundary(SubDomain):
    def inside(self, x, on_bnd):
        return norm(x-np.array([ d,0]))<r+EPS and on_bnd

outer_bnd = OuterBoundary()
left_bnd = LeftObjectBoundary()
right_bnd = RightObjectBoundary()

res = resolutions[-1]
order = orders[-1]+1
print("Solving res {}, order {} (reference)".format(res,order))

mesh = generate_mesh(domain, res_start)
plot(mesh); plt.show()
for res in resolutions[:-1]:
    mesh = refine(mesh)

bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
bnd.set_all(0)
outer_bnd.mark(bnd, 1)
left_bnd.mark( bnd, 2)
right_bnd.mark(bnd, 3)

V = FunctionSpace(mesh, "Lagrange", order)

phi = TrialFunction(V)
psi = TestFunction(V)

bcs = [DirichletBC(V, Constant(i), bnd, i+1) for i in range(3)]

lhs = dot(grad(phi), grad(psi)) * dx
rhs = rho*psi * dx

A = assemble(lhs)
b = assemble(rhs)

for bc in bcs:
    bc.apply(A, b)

phi_ref = Function(V)
phi_ref.set_allow_extrapolation(allow_extrapolation)

solver = PETScKrylovSolver('gmres','hypre_amg')
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10 #e-12
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = monitor_convergence

solver.set_operator(A)
solver.solve(phi_ref.vector(), b)

hmins = {}
errors = {}
mesh = generate_mesh(domain, res_start)

for res in resolutions:

    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, 1)
    left_bnd.mark( bnd, 2)
    right_bnd.mark(bnd, 3)

    errors[res] = {}
    hmins[res] = mesh.hmin()
    for order in orders:

        print("Solving res {}, order {}".format(res,order))

        V = FunctionSpace(mesh, "Lagrange", order)

        phi = TrialFunction(V)
        psi = TestFunction(V)

        bcs = [DirichletBC(V, Constant(i), bnd, i+1) for i in range(3)]

        lhs = dot(grad(phi), grad(psi)) * dx
        rhs = rho*psi * dx

        A = assemble(lhs)
        b = assemble(rhs)

        for bc in bcs:
            bc.apply(A, b)

        phi_sol = Function(V)
        phi_sol.set_allow_extrapolation(allow_extrapolation)

        solver = PETScKrylovSolver('gmres','hypre_amg')
        solver.parameters['absolute_tolerance'] = 1e-14
        solver.parameters['relative_tolerance'] = 1e-10 #e-12
        solver.parameters['maximum_iterations'] = 100000
        solver.parameters['monitor_convergence'] = monitor_convergence

        solver.set_operator(A)
        solver.solve(phi_sol.vector(), b)

        error = errornorm(phi_ref, phi_sol, degree_rise=0)
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

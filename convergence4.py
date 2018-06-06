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

set_log_level(WARNING)

orders              = [1,2]
resolutions         = [1,2,3,4,5]
method = 'bicgstab'
precond = 'ilu'
monitor = True
use_dirichlet = True
segments = 70

rho = Expression("100*x[1]", degree=1)
# rho = Constant(0)

# l = 0.2
# ll = 1.0
# lh = 0.05
# lo = 0.7
# li = 0.1
# EPS = DOLFIN_EPS

# domain = Rectangle(Point(-ll,-ll), Point(ll,ll)) \
#        - Circle(Point(0,0), l, 90)

r = 1
d = 3*r
h = 3*r
l = d+3*r
EPS = 1e-2 #DOLFIN_EPS

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

mesh = generate_mesh(domain, 10)
print(len(list(cells(mesh))))
plot(mesh); plt.show()

for res in resolutions:
    mesh = refine(mesh)

order = 3
print("Order: {}, resolution: {} (reference)".format(order,res))

bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
bnd.set_all(0)
outer_bnd.mark(bnd, 1)
left_bnd.mark( bnd, 2)
right_bnd.mark(bnd, 3)

V = FunctionSpace(mesh, 'CG', order)
phi = TrialFunction(V)
psi = TestFunction(V)

a = dot(grad(phi), grad(psi)) * dx
L = rho*psi * dx

bcs = [DirichletBC(V, Constant(i), bnd, i+1) for i in range(3)]
phi_ref = Function(V)

# solve(a==L, phi, bcs=bc)

A = assemble(a)
b = assemble(L)
for bc in bcs:
    bc.apply(A, b)

solver = PETScKrylovSolver(method, precond)
solver.parameters['absolute_tolerance'] = 1e-14
solver.parameters['relative_tolerance'] = 1e-10
solver.parameters['maximum_iterations'] = 100000
solver.parameters['monitor_convergence'] = monitor
solver.parameters['nonzero_initial_guess'] = True

solver.set_operator(A)
solver.solve(phi_ref.vector(), b)

File('phi_ref.pvd') << phi_ref

n = FacetNormal(mesh)
dss = Measure("ds", domain=mesh, subdomain_data=bnd)

charge1 = assemble(dot(grad(phi_ref), n) * dss(2, degree=3))
charge2 = assemble(dot(grad(phi_ref), n) * dss(3, degree=3))
total_charge = charge1 + charge2

mesh = generate_mesh(domain, 10)

hmins = {}
errors = {}
for res in resolutions:

    bnd = MeshFunction('size_t', mesh, mesh.geometry().dim()-1)
    bnd.set_all(0)
    outer_bnd.mark(bnd, 1)
    left_bnd.mark( bnd, 2)
    right_bnd.mark(bnd, 3)

    hmins[res] = mesh.hmin()
    errors[res] = {}
    for order in orders:

        print("Order: {}, resolution: {}".format(order,res))

        V = FunctionSpace(mesh, 'CG', order)
        phi = TrialFunction(V)
        psi = TestFunction(V)

        a = dot(grad(phi), grad(psi)) * dx
        L = rho*psi * dx

        vsources = [[1,0,-1]]

        bce = DirichletBC(V, Constant(0), bnd, 1)
        objects = [ObjectBC(V, bnd, 2+i) for i in range(2)]
        circuit = Circuit(V, bnd, objects, vsources)
        objects[0].charge = total_charge
        bcs = [DirichletBC(V, Constant(i), bnd, i+1) for i in range(3)]
        phi_sol = Function(V)
        phi_sol.vector()[:] = 1

        # solve(a==L, phi, bcs=bc)

        A = assemble(a)
        b = assemble(L)

        if use_dirichlet:
            for bc in bcs:
                bc.apply(A, b)
        else:
            bce.apply(A, b)
            A, b = circuit.apply(A, b)
            for o in objects:
                o.apply(A, b)

        solver = PETScKrylovSolver(method,precond)
        solver.parameters['absolute_tolerance'] = 1e-14
        solver.parameters['relative_tolerance'] = 1e-10 #e-12
        solver.parameters['maximum_iterations'] = 100000
        solver.parameters['monitor_convergence'] = monitor

        solver.set_operator(A)
        solver.solve(phi_sol.vector(), b)

        File('phi_order{}.pvd'.format(order)) << phi_sol

        error = errornorm(phi_ref, phi_sol, degree_rise=0, mesh=mesh)
        # error = errornorm(phi_ref, phi_sol, degree_rise=0)
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

    with open('convergence4.txt', 'a') as file:
        for xx,yy in zip(x,y):
            file.write("{} {} {} {} {}\n".format(use_dirichlet, order, segments, xx, yy))

plt.grid()
plt.xlabel('$h_\\mathrm{{min}}$')
plt.ylabel('$L_2$ norm error')
plt.title('Convergence')
plt.legend(loc='lower right')
plt.show()
